
import math
import pymol
import numpy as np

from gen_ensemble.utils.dihedral_ops import *

def dihedral_distance_per_angle_pair(theta1, theta2):

    return (2.0 * (1.0 - math.cos(math.radians(theta1) - math.radians(theta2))))

def compute_diff_array(p1, p2, ignore=[]):

    diff_arr = []
    if len(p1) == len(p2):
        for i in range(0, len(p1)):
            if (p1[i] is not None) and (p2[i] is not None):
                if i not in ignore:
                    diff_arr.append(round(dihedral_distance_per_angle_pair(p1[i], p2[i]), 3))

    return diff_arr

def distance(p1, p2, ignore = []):

    phi_diff_arr = []
    psi_diff_arr = []
    (phi_p1, psi_p1, omega_p1) = get_phi_psi_omega_angles(p1)
    (phi_p2, psi_p2, omega_p2) = get_phi_psi_omega_angles(p2)

    phi_diff_arr = compute_diff_array(phi_p1, phi_p2, ignore)
    psi_diff_arr = compute_diff_array(psi_p1, psi_p2, ignore)
    omega_diff_arr = compute_diff_array(omega_p1, omega_p2, ignore)

    return (phi_diff_arr, psi_diff_arr, omega_diff_arr)

def average_distance(p1, p2, ignore = []):

    (phi_diff_arr, psi_diff_arr, omega_diff_arr) = distance(p1, p2, ignore)
    dihedral_diff_arr = phi_diff_arr+psi_diff_arr+omega_diff_arr

    if len(phi_diff_arr) > 0 and len(psi_diff_arr) > 0 and len(dihedral_diff_arr) > 0 and len(omega_diff_arr) > 0:
        return (np.mean(phi_diff_arr), np.mean(psi_diff_arr), np.mean(dihedral_diff_arr), np.mean(omega_diff_arr))
    else:
        return (None, None, None, None)

def median_distance(p1, p2, ignore = []):

    (phi_diff_arr, psi_diff_arr, omega_diff_arr) = distance(p1, p2, ignore)
    dihedral_diff_arr = phi_diff_arr+psi_diff_arr+omega_diff_arr

    if len(phi_diff_arr) > 0 and len(psi_diff_arr) > 0 and len(dihedral_diff_arr) > 0 and len(omega_diff_arr) > 0:
        return (np.median(phi_diff_arr), np.median(psi_diff_arr), np.median(dihedral_diff_arr), np.median(omega_diff_arr))
    else:
        return (None, None, None, None)

def max_distance(p1, p2, ignore = []):

    (phi_diff_arr, psi_diff_arr, omega_diff_arr) = distance(p1, p2, ignore)
    dihedral_diff_arr = phi_diff_arr+psi_diff_arr+omega_diff_arr

    if len(phi_diff_arr) > 0 and len(psi_diff_arr) > 0 and len(dihedral_diff_arr) > 0 and len(omega_diff_arr) > 0:
        return (max(phi_diff_arr), max(psi_diff_arr), max(dihedral_diff_arr), max(omega_diff_arr))
    else:
        return (None, None, None, None)

def is_any_angle_violated(diff_arr, violation_thresh):

    violated = False
    for i in range(0, len(diff_arr)):
        if diff_arr[i] > violation_thresh:
            violated = True
            print ("Violated angle by residue id: ", i+1, " by ", diff_arr[i])

    return violated

def get_phi_psi_omega_pymol(pdb, chain_id, pep_residues_for_looping):

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

def distance_pymol(pdb1, pdb2, chain1, chain2, pep1, pep2):

    phi_diff_arr = []
    psi_diff_arr = []
    omega_diff_arr = []
    (phi_p1, psi_p1, omega_p1) = get_phi_psi_omega_pymol(pdb1, chain1, pep1)
    (phi_p2, psi_p2, omega_p2) = get_phi_psi_omega_pymol(pdb2, chain2, pep2)

    phi_diff_arr = compute_diff_array(phi_p1, phi_p2, [])
    psi_diff_arr = compute_diff_array(psi_p1, psi_p2, [])
    omega_diff_arr = compute_diff_array(omega_p1, omega_p2, [])

    return (phi_diff_arr, psi_diff_arr, omega_diff_arr)

def d_score(pdb1, pdb2, chain1, chain2, pep1, pep2):

    (phi_diff_arr, psi_diff_arr, omega_diff_arr) = distance_pymol(pdb1, pdb2, chain1, chain2, pep1, pep2)
    dscore = sum(phi_diff_arr) + sum(psi_diff_arr) + sum(omega_diff_arr)
    return dscore

def d_score_no_omega(pdb1, pdb2, chain1, chain2, pep1, pep2):

    (phi_diff_arr, psi_diff_arr, omega_diff_arr) = distance_pymol(pdb1, pdb2, chain1, chain2, pep1, pep2)
    dscore = sum(phi_diff_arr) + sum(psi_diff_arr)
    return dscore