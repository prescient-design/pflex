"""
Utility functions to read PDB files, from MD or FlexPepDock, and convert to
tensors for downstream analysis

Author: Darcy Davidson
Minor modifications by Ji Won Park

"""
import glob
import boto3
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import pymol
import numpy as np
import os
from MDAnalysis.analysis import align


def download_file_from_s3(bucket_name, key, local_dir):
    """
    Download a named file from an S3 bucket to a local directory

    """
    s3 = boto3.client('s3')
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_file_path = os.path.join(local_dir, os.path.basename(key))
    s3.download_file(bucket_name, key, local_file_path)


def download_pdb_files_from_s3(
        bucket_name, s3_prefix, local_dir):
    s3 = boto3.client('s3')
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # List objects within the specified S3 prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".pdb"):
            local_file_path = os.path.join(local_dir, os.path.basename(key))
            s3.download_file(bucket_name, key, local_file_path)
            print(f"Downloaded {key} to {local_file_path}")


def sort_atoms(atoms):
    # Sort atoms by residue number and atom name
    sorted_atoms = atoms.atoms[np.lexsort(
        (atoms.names, atoms.resnums, atoms.segids))]
    return sorted_atoms


def print_sorted_atoms(sorted_atoms, num_lines=1556):
    print(
        f"{'Atom':<6} {'ResNum':<6} {'ResName':<6} {'Name':<6}"
        f" {'X':<8} {'Y':<8} {'Z':<8}")
    for i, atom in enumerate(sorted_atoms[:num_lines]):
        print(
            f"{atom.id:<6} {atom.resnum:<6} {atom.resname:<6} {atom.name:<6}"
            f" {atom.position[0]:<8.3f} {atom.position[1]:<8.3f}"
            f" {atom.position[2]:<8.3f}")


def align_models_within_ensemble(ensemble_pdb, output_pdb):
    u = mda.Universe(ensemble_pdb)
    ref_atoms = u.select_atoms('backbone and not element H')

    # Align all frames to the first frame
    aligner = align.AlignTraj(
        u, u, select='backbone and not element H', in_memory=True).run()

    # Write the aligned ensemble to a new PDB file
    u.atoms.write(output_pdb)
    # print(
    #     f"Aligned all models within {ensemble_pdb} to the first model and "
    #     f" wrote to {output_pdb}")


def md_pdb_to_tensor_no_h(md_file_path):
    u = mda.Universe(md_file_path)
    models = []
    for ts in u.trajectory:
        # Select all atoms except Na+ and hydrogens
        atoms = u.select_atoms("not resname Na+ and not element H")
        sorted_atoms = sort_atoms(atoms)
        coords = sorted_atoms.positions
        models.append(coords)

    # Print the first 40 lines of the sorted data
    # print_sorted_atoms(sorted_atoms)
    tensor = np.array(models)
    return tensor


def get_phi_psi_omega_pymol(pdb, pep_residues_for_looping, chain_id="B"):

    phi_arr = []
    psi_arr = []
    omega_arr = []

    pymol.cmd.load(pdb, "obj01")

    current = pep_residues_for_looping[0]
    residue_def = "resi " + str(current)
    residue_def_next = "resi " + str(current+1)

    try:

        psi_first = [# Psi angles
        "/obj01//"+chain_id+"/"+str(current)+'/N',
        "/obj01//"+chain_id+"/"+str(current)+'/CA',
        "/obj01//"+chain_id+"/"+str(current)+'/C',
        "/obj01//"+chain_id+"/"+str(current+1)+'/N']

        psi = pymol.cmd.dihedral('dih',psi_first[0],psi_first[1],psi_first[2],psi_first[3])

        phi_arr.append(None)
        psi_arr.append(psi)
        omega_arr.append(None)

        start = pep_residues_for_looping[0]
        for i in range(1, len(pep_residues_for_looping)-1):

            # Define selections for residue i-1, i and i+1
            residue_def = 'resi '+str(start+i)
            residue_def_prev = 'resi '+str(start+i-1)
            residue_def_next = 'resi '+str(start+i+1)

            phi_psi = [
                    # Phi angles
                    "/obj01//"+chain_id+"/"+str(start+i-1)+'/C',
                    "/obj01//"+chain_id+"/"+str(start+i)+'/N',
                    "/obj01//"+chain_id+"/"+str(start+i)+'/CA',
                    "/obj01//"+chain_id+"/"+str(start+i)+'/C',
                    # Psi angles
                    "/obj01//"+chain_id+"/"+str(start+i)+'/N',
                    "/obj01//"+chain_id+"/"+str(start+i)+'/CA',
                    "/obj01//"+chain_id+"/"+str(start+i)+'/C',
                    "/obj01//"+chain_id+"/"+str(start+i+1)+'/N',
                    # Omega angles
                    "/obj01//"+chain_id+"/"+str(start-i)+'/CA',
                    "/obj01//"+chain_id+"/"+str(start-i)+'/C',
                    "/obj01//"+chain_id+"/"+str(start)+'/N',
                    "/obj01//"+chain_id+"/"+str(start)+'/CA']

            # Compute phi/psi angle
            phi = pymol.cmd.dihedral('dih', phi_psi[0],phi_psi[1],phi_psi[2],phi_psi[3])
            psi = pymol.cmd.dihedral('dih', phi_psi[4],phi_psi[5],phi_psi[6],phi_psi[7])
            omega = pymol.cmd.dihedral('dih', phi_psi[8],phi_psi[9],phi_psi[10],phi_psi[11])

            phi_arr.append(phi)
            psi_arr.append(psi)
            omega_arr.append(omega)

        current = pep_residues_for_looping[len(pep_residues_for_looping)-1]
        residue_def = "resi " + str(current)
        residue_def_prev = "resi " + str(current-1)

        phi_last = [# Phi angles
        "/obj01//"+chain_id+"/"+str(current-1)+'/C',
        "/obj01//"+chain_id+"/"+str(current)+'/N',
        "/obj01//"+chain_id+"/"+str(current)+'/CA',
        "/obj01//"+chain_id+"/"+str(current)+'/C']

        omega_last = [# Omega angles
        "/obj01//"+chain_id+"/"+str(current-1)+'/CA',
        "/obj01//"+chain_id+"/"+str(current-1)+'/C',
        "/obj01//"+chain_id+"/"+str(current)+'/N',
        "/obj01//"+chain_id+"/"+str(current)+'/CA']

        phi = pymol.cmd.dihedral("dih", phi_last[0],phi_last[1],phi_last[2],phi_last[3])
        omega = pymol.cmd.dihedral("dih", omega_last[0],omega_last[1],omega_last[2],omega_last[3])

        phi_arr.append(phi)
        psi_arr.append(None)
        omega_arr.append(omega)

    except pymol.CmdException as e:
        err = "More than one atom found"
        if err in str(e):
            return ([], [], [])


    pymol.cmd.delete("all")

    return (phi_arr, psi_arr, omega_arr)


def pdb_to_tensor(pdb_file):
    u = mda.Universe(pdb_file)

    # Select atoms excluding hydrogens
    non_hydrogen_atoms = u.select_atoms("not element H")
    sorted_atoms = sort_atoms(non_hydrogen_atoms)

    # Print the first 40 lines of the sorted data
    # print_sorted_atoms(sorted_atoms)

    # Extract the atomic coordinates
    coordinates = sorted_atoms.positions

    # Convert coordinates to a tensor (numpy array)
    tensor = np.array(coordinates)

    return tensor


def write_pdb_without_hydrogens(input_pdb_file, output_pdb_file):
    u = mda.Universe(input_pdb_file)

    # Select atoms excluding hydrogens
    non_hydrogen_atoms = u.select_atoms('not element H')
    sorted_atoms = sort_atoms(non_hydrogen_atoms)

    # Write the new PDB file
    sorted_atoms.write(output_pdb_file)
    # print(f"Wrote {output_pdb_file} without hydrogens")


def align_pdb_files(reference_pdb, target_pdb, output_pdb):
    ref = mda.Universe(reference_pdb)
    target = mda.Universe(target_pdb)

    # Select backbone atoms for alignment
    # ref_atoms = ref.select_atoms('backbone and not element H')
    # target_atoms = target.select_atoms('backbone and not element H')

    # Perform the alignment
    aligner = align.AlignTraj(
        target, ref, select='backbone and not element H', in_memory=True)
    aligner.run()

    # Write the aligned target structure to a new PDB file
    target.atoms.write(output_pdb)
    # print(f"Aligned {target_pdb} to {reference_pdb} and wrote to {output_pdb}")
