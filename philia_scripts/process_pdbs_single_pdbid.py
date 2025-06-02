"""
Script to download PDB files from S3 corresponding to a PDB ID and align the
atoms between the MD frames and the FPD-generated frames

Author: Darcy Davidson
Modified by Ji Won Park

Usage
-----
python scripts/process_pdbs.py --pdbid=3KLA \
--local_dir="/scratch/site/u/parj2/pmhc_pdb_files" --skip_download

Directory structure
-------------------

local_dir
    |_ pdbid_dir
        |_ set0
            |_ aligned
        |_ set1
            |_ aligned
        |_ set2
            |_ aligned
        |_ set3
            |_ aligned
        |_ set4
            |_ aligned
        |_ md
            |_ aligned

"""

import argparse
import os
import glob
from philia.analysis import io


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdbid", type=str,
        help="PDB ID")
    parser.add_argument(
        "--local_dir", type=str,
        default="/scratch/site/u/parj2/pmhc_pdb_files",
        help="Local directory to store downloaded PDB files")
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Whether to skip download")
    args = parser.parse_args()

    # Set the local directory to the input directory provided through
    # command-line args
    pdbid_dir = os.path.join(args.local_dir, args.pdbid)
    os.makedirs(pdbid_dir, exist_ok=True)
    pdbid_md_dir = os.path.join(pdbid_dir, "md")

    # Download PDB files from S3
    bucket_name = "prescient-data-dev"  # S3 bucket name
    if not args.skip_download:
        # Download FPD predictions for all 5 sets
        for set_idx in range(5):
            pdbid_set_dir = os.path.join(pdbid_dir, f"set{set_idx}")
            s3_prefix = f"sandbox/nerlis/scratch/pmhc/v1_06172024/set{set_idx}/runs_0.1/{args.pdbid}/"
            io.download_pdb_files_from_s3(bucket_name, s3_prefix, pdbid_set_dir)
    # Download MD
    io.download_pdb_files_from_s3(
        bucket_name,
        f"raw/MD-simulations/tcr_pMHC/md-pdb-ensembles/{args.pdbid}_reordered_ensemble.pdb",
        pdbid_md_dir)

    # Align all models within {pdbid}_reordered_ensemble.pdb to the first model
    ensemble_pdb = os.path.join(
        pdbid_md_dir,
        f"{args.pdbid}_reordered_ensemble.pdb")
    aligned_ensemble_pdb = os.path.join(
        pdbid_md_dir,
        f"aligned_{args.pdbid}_reordered_ensemble.pdb")
    if os.path.exists(ensemble_pdb):
        io.align_models_within_ensemble(ensemble_pdb, aligned_ensemble_pdb)

    # Read each PDB file, align to the first model of
    # {pdbid}_reordered_ensemble.pdb, convert to tensor, and write new PDB
    # without hydrogens
    failed_pdb_files = []
    for pdb_file in glob.glob(os.path.join(pdbid_dir, "set*", "*_refined_*.pdb")):
        # Align the PDB file to the first model of 3KLA_reordered_ensemble.pdb
        pdbid_set_aligned_dir = os.path.join(
            os.path.dirname(pdb_file), "aligned")
        os.makedirs(pdbid_set_aligned_dir, exist_ok=True)
        aligned_pdb_file = os.path.join(pdbid_set_aligned_dir,
            f"aligned_{os.path.basename(pdb_file)}")
        try:
            io.align_pdb_files(aligned_ensemble_pdb, pdb_file, aligned_pdb_file)
        except:
            print("Failed to align", pdb_file)
            failed_pdb_files.append(pdb_file)
            continue

    print("Failed to align the following PDB files: ", failed_pdb_files)

    # # Example usage for MD trajectory
    # output_md_pdb = os.path.join(
    #     pdbid_md_dir,
    #     f"no_h_{args.pdbid}_reordered_ensemble.pdb")
    # tensor_md = io.md_pdb_to_tensor(ensemble_pdb, output_md_pdb)
    # print(tensor_md.shape)  # Should print (300, num_atoms, 3)
