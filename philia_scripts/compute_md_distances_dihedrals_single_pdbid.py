"""
Script to compute distributional distances between the MD and FPD-generated
frames for a single PDB ID, in the dihedrals space.

"""
import os
import time
from math import lcm
import glob
import numpy as np
import pandas as pd
import torch
from calibration.distances.wasserstein_sliced import (
    get_gaussian_sliced_p_wasserstein_from_samples)
import calibration.distances.mmd as mmd_utils
from philia.analysis import io
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute distributional distances between MD and"
        " FPD-generated frames for a single PDB ID")
    parser.add_argument("--pdbid", type=str, default="1BD2", help="PDB ID")
    parser.add_argument(
        "--include_omega", default=False,
        action="store_true",
        help="Exclude omega angle")
    parser.add_argument(
        "--standardize", default=False,
        action="store_true",
        help="Standardize")
    parser.add_argument(
        "--local_dir_tensors", type=str,
        default="/scratch/site/u/parj2/pmhc_tensors",
        help="Local directory to store tensors")
    args = parser.parse_args()

    pdbid = args.pdbid
    local_dir = "/scratch/site/u/parj2/pmhc_tensors"  # args.local_dir_tensors

    # # Check if the output distances exist
    # df = pd.read_csv(f"/scratch/site/u/parj2/pmhc_fpd_files/rosetta_score_overlapping_15.0.csv")
    # pdbid_list = np.sort(df["pdbid"].unique())  # 141 PDB IDs
    # for pdbid_check in pdbid_list:
    #     for dscore in ["2.5", "6.0", "10.0", "15.0"]:
    #         out_dir = os.path.join(args.local_dir_tensors, pdbid_check, dscore)
    #         out_dist_path = os.path.join(out_dir, "distances.npy")
    #         if not os.path.exists(out_dist_path):
    #             print(f"Missing {out_dist_path}")
    # breakpoint()

    for dscore in ["10.0"]:
        fpd_path = os.path.join(local_dir, pdbid, dscore, "fpd_tensor.npy")
        md_path = os.path.join(local_dir, pdbid, dscore, "md_tensor.npy")
        is_included_path = os.path.join(local_dir, pdbid, dscore, "is_included_mask.npy")
        # breakpoint()
        fpd_tensor = np.load(fpd_path)
        print(f"FPD tensor shape: {fpd_tensor.shape}")
        mask = np.load(is_included_path)
        # fpd_tensor = fpd_tensor[mask]  # [num_samples, pep_len-1, 3]
        md_tensor = np.load(md_path)  # [300, pep_len-1, 3]

        # Convert to rad
        fpd_tensor = np.deg2rad(fpd_tensor)
        md_tensor = np.deg2rad(md_tensor)

        print(f"FPD tensor shape: {fpd_tensor.shape}")
        print(f"Mask tensor shape: {mask.shape}")
        print(f"MD tensor shape: {md_tensor.shape}")

        num_angles = 3
        if not args.include_omega:
            print("Excluding omega angle")
            fpd_tensor = fpd_tensor[:, :, :2]
            md_tensor = md_tensor[:, :, :2]
            num_angles = 2

        if args.standardize:
            meta_mean = np.mean(fpd_tensor, axis=0, keepdims=True)  # [pep_len-1, 2]
            meta_std = np.std(fpd_tensor, axis=0, keepdims=True)  # [pep_len-1, 2]
            fpd_tensor = (fpd_tensor - meta_mean)/meta_std
            md_tensor = (md_tensor - meta_mean)/meta_std

        # Reshape to [num_samples, (L-1)*3]
        num_fpd, pep_len_minus_1, _ = fpd_tensor.shape
        num_md = md_tensor.shape[0]
        dim = pep_len_minus_1*num_angles
        fpd_tensor = fpd_tensor.reshape(num_fpd, dim)
        md_tensor = md_tensor.reshape(num_md, dim)

        # start_dist = time.time()
        dist = {}
        for gamma in [0.1, 1.0, 10.0]:
            kernel_fn = mmd_utils.rbf_kernel_wrapped
            dist[f"mmd_rbf_gamma_{gamma:.2f}"] = mmd_utils.get_mmd(
                torch.tensor(fpd_tensor), torch.tensor(md_tensor),
                kernel_fn=kernel_fn,
                gamma=gamma)
        dist[f"mmd_cos"] = mmd_utils.pair_mmd_cos_distance(
            torch.tensor(fpd_tensor), torch.tensor(md_tensor))
        dist[f"mmd_dot"] = mmd_utils.pair_mmd_dot_distance(
            torch.tensor(fpd_tensor), torch.tensor(md_tensor))
        for temp in [1.e-6, 1e-8, 1.e-10]:
            dist[f"hyperbolic_rbf_mmd_temp_{temp:.4f}"] = mmd_utils.hyperbolic_rbf_mmd(
                torch.tensor(fpd_tensor),
                torch.tensor(md_tensor),
                temp=temp)
            # print(dist[f"hyperbolic_rbf_mmd_temp_{temp:.4f}"])

        num_md_samples = md_tensor.shape[0]
        num_proj = 100000
        print(f"Number of projections: {num_proj}")
        num_subsample = 64
        w2_subsamples = np.zeros(num_subsample)
        for subsample_i in range(num_subsample):
            subsample_idx = np.random.choice(num_fpd, num_md_samples, replace=False)
            subsample_fpd = fpd_tensor[subsample_idx]
            w2_subsamples[subsample_i] = get_gaussian_sliced_p_wasserstein_from_samples(
                torch.tensor(subsample_fpd), torch.tensor(md_tensor),
            num_projections=num_proj)

        dist["w2"] = w2_subsamples.mean()
        dist["w2_full"] = w2_subsamples
        # # Inflate the tensors to have the same number of samples
        # num_samples = lcm(num_fpd, num_md)
        # print(f"Number of samples: {num_samples}")
        # inflated_pred_tensor = np.repeat(
        #     fpd_tensor, num_samples // num_fpd, axis=0).reshape(num_samples, -1)
        # inflated_md_tensor = np.repeat(
        #     md_tensor, num_samples // num_md, axis=0).reshape(num_samples, -1)
        # # both [num_samples, num_heavy_atoms*3]
        # if num_samples > 10000:
        #     num_proj = 1000
        # else:
        #     num_proj = 10000

        # end_dist = time.time()
        # print("Time to compute distances: ", end_dist - start_dist)
        if args.standardize:
            out_dict_path = os.path.join(local_dir, pdbid, dscore, "distances_standardized.npy")
        else:
            out_dict_path = os.path.join(local_dir, pdbid, dscore, "distances.npy")

        print(f"Final distances: {dist}")

        np.save(out_dict_path, dist, allow_pickle=True)
        print("Saved distances to ", out_dict_path)


