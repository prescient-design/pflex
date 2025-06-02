"""
Script to compute distributional distances between the MD and FPD-generated
frames for a single PDB ID, in the dihedrals space.

"""
import os
import sys
import time
from math import lcm
from tqdm import tqdm
import glob
import numpy as np
import pandas as pd
import torch
from calibration.distances.wasserstein_sliced import (
    get_gaussian_sliced_p_wasserstein_from_samples)
from calibration.distances.mmd import rbf_mmd
from philia.analysis import io
import argparse
# import MDAnalysis as mda
# from MDAnalysis.analysis.dihedrals import Dihedral


def remove_terminal_nans(phi, psi, omega):
    return phi[1:], psi[:-1], omega[1:]


def write_dihedrals(pdbid, local_dir_fpd, local_dir_md, local_dir_tensors):
    print(f"Processing {pdbid}")
    is_success = True  # init
    # Peptide length determines tensor shape
    pep_len = df[df["pdbid"] == pdbid]["pep_len"].iloc[0]

    # FPD paths corresponding to this pdbid, there should be 200 of them
    fpd_paths_by_dscore = {}
    is_included_mask_by_dscore = {}
    for dscore in ["10.0"]:  # ["2.5", "6.0", "10.0", "15.0"]:
        overlapping_csv_path = os.path.join(
            local_dir_fpd, f"rosetta_score_overlapping_{dscore}.csv")
        if not os.path.exists(overlapping_csv_path):
            overlapping_csv_path = os.path.join(
            local_dir_fpd, f"rosetta_score_overlapping.csv")
        dscore_df = pd.read_csv(overlapping_csv_path)
        # Get rows corresponding to this pdbid
        rows_pdbid = dscore_df[dscore_df["pdbid"] == pdbid]
        # Get unique tags
        tags = rows_pdbid["tag"].values
        is_included = (rows_pdbid["intersect"].values == "Overlapping")
        tags, unique_idx = np.unique(tags, return_index=True)
        is_included = is_included[unique_idx]
        # Sort tags
        tags_idx = np.argsort(tags)
        tags = tags[tags_idx]
        # Sort is_included values them so they align with tags
        is_included = is_included[tags_idx]

        fpd_paths = []
        is_included_mask = []  # index of the tag out of the total 1000 rows
        print(f"Number of rows for {pdbid}: {len(rows_pdbid)}")
        for i, tag in enumerate(tags):
            match = glob.glob(f"{local_dir_fpd}/set*/runs_0.1/{pdbid}/{tag.split('_set')[0]}.pdb")
            if len(match) == 0:
                match = glob.glob(f"{local_dir_fpd}/set*/runs_0.1/{pdbid}/*/{tag.split('_set')[0]}.pdb")
            if len(match) > 0:
                fpd_paths.append(match[0])
                is_included_mask.append(is_included[i])
        print(f"Number of matches for {pdbid}: {len(fpd_paths)}")
        if (len(rows_pdbid) != len(fpd_paths)):
            is_success = False
        fpd_paths_by_dscore[dscore] = np.array(fpd_paths)
        is_included_mask_by_dscore[dscore] = np.array(is_included_mask)

        # Read into tensor of shape [200, L-1, 3]
        fpd_tensor = np.empty((len(fpd_paths_by_dscore[dscore]), pep_len - 1, 3))
        for i, path in enumerate(fpd_paths_by_dscore[dscore]):
            pep_residues_for_looping = [i for i in range(1, pep_len+1)]
            phi, psi, omega = io.get_phi_psi_omega_pymol(path, pep_residues_for_looping)
            phi, psi, omega = remove_terminal_nans(phi, psi, omega)
            fpd_tensor[i] = np.stack([phi, psi, omega], axis=1)

        # MD paths corresponding to this pdbid, there should be 300 of them
        md_paths = glob.glob(f"{local_dir_md}/{pdbid}/*no_na.cleaned.pdb")
        md_paths = np.sort(md_paths)

        # Read into tensor of shape [300, L-1, 3]
        md_tensor = np.empty((len(md_paths), pep_len - 1, 3))
        for i, path in enumerate(md_paths):
            pep_residues_for_looping = [i for i in range(1, pep_len+1)]
            phi, psi, omega = io.get_phi_psi_omega_pymol(path, pep_residues_for_looping)
            phi, psi, omega = remove_terminal_nans(phi, psi, omega)
            md_tensor[i] = np.stack([phi, psi, omega], axis=1)

        # Make the output directory if it doesn't exist
        out_dir = os.path.join(local_dir_tensors, pdbid, dscore)
        os.makedirs(out_dir, exist_ok=True)

        # Write FPD and MD tensors
        np.save(os.path.join(out_dir, "fpd_tensor.npy"), fpd_tensor)
        np.save(os.path.join(out_dir, "md_tensor.npy"), md_tensor)
        np.save(os.path.join(out_dir, "is_included_mask.npy"),
                is_included_mask_by_dscore[dscore])
    return is_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute distributional distances between MD and"
        " FPD-generated frames for a single PDB ID")
    parser.add_argument(
        "--local_dir_md", type=str,
        default="/scratch/site/u/parj2/pmhc_md_files",
        help="Local directory that stores downloaded PDB files from MD")
    parser.add_argument(
        "--local_dir_tensors", type=str,
        default="/scratch/site/u/parj2/pmhc_tensors",
        help="Local directory to store tensors")
    args = parser.parse_args()

    local_dir_md = args.local_dir_md
    os.makedirs(args.local_dir_tensors, exist_ok=True)

    local_dir_fpd = "/scratch/site/u/parj2/pmhc_fpd_files"
    # Load FPD structures that pass Rosetta filters and
    # dscore < dscore_cutoff
    df = pd.read_csv(
        glob.glob(os.path.join(local_dir_fpd, f"rosetta_score_overlapping*.csv"))[0])

    pdbid_list = np.sort(df["pdbid"].unique())  # 141 PDB IDs
    pdbids_8_9mer = np.sort(df[df["pep_len"] < 10]["pdbid"].unique())
    pdbids_10mer = np.sort(df[df["pep_len"] == 10]["pdbid"].unique())
    pdbids_11mer = np.sort(df[df["pep_len"] == 11]["pdbid"].unique())
    pdbids_12_13mer = np.sort(df[df["pep_len"] > 11]["pdbid"].unique())
    assert len(pdbid_list) == len(pdbids_8_9mer) + len(pdbids_10mer) + len(pdbids_11mer) + len(pdbids_12_13mer)

    # # Print pdbids_8_9mer as a bash list
    # print("pdbids_8_9mer=(", " ".join(pdbids_8_9mer), ")")
    # # Print pdbids_10mer as a bash list
    # print("pdbids_10mer=(", " ".join(pdbids_10mer), ")")
    # # Print pdbids_11mer as a bash list
    # print("pdbids_11mer=(", " ".join(pdbids_11mer), ")")
    # # Print pdbids_12_13mer as a bash list
    # print("pdbids_12_13mer=(", " ".join(pdbids_12_13mer), ")")

    # success_list = []
    # for pdbid in tqdm(pdbids_8_9mer, desc="8_9mer"):
    #     local_dir_fpd = "/scratch/site/u/parj2/pmhc_fpd_files"
    #     is_success = write_dihedrals(pdbid, local_dir_fpd, args.local_dir_md, args.local_dir_tensors)
    #     success_list.append(is_success)
    # success_dict = dict(zip(pdbids_8_9mer, success_list))
    # print(f"Failed pdbids: {np.array(pdbids_8_9mer)[~np.array(success_list)]}")
    # np.save("success_dict_8_9mer.npy", success_dict)

    success_list = []
    for pdbid in tqdm(pdbids_11mer, desc="11mer"):
        local_dir_fpd = "/scratch/site/u/parj2/v1_multi"
        is_success = write_dihedrals(pdbid, local_dir_fpd, args.local_dir_md, args.local_dir_tensors)
        success_list.append(is_success)
    success_dict = dict(zip(pdbids_11mer, success_list))
    print(f"Failed pdbids: {np.array(pdbids_11mer)[~np.array(success_list)]}")
    np.save("success_dict_11mer.npy", success_dict)

