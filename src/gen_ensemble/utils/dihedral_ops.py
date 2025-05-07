
import math
import hydra
from Bio.PDB import *
from omegaconf import DictConfig
from Bio.PDB import internal_coords


def get_phi_psi_angles(chain):

    phi = []
    psi = []
    chain.atom_to_internal_coordinates(verbose=False)
    for residue in chain:
        ric = residue.internal_coord
        if ric:
            phi.append(ric.get_angle("phi"))
            psi.append(ric.get_angle("psi"))

    return (phi, psi)

def get_phi_psi_omega_angles(chain):

    phi = []
    psi = []
    omega = []
    chain.atom_to_internal_coordinates(verbose=False)
    for residue in chain:
        ric = residue.internal_coord
        if ric:
            phi.append(ric.get_angle("phi"))
            psi.append(ric.get_angle("psi"))
            omega.append(ric.get_angle("omg"))

    return (phi, psi, omega)

def get_phi_psi_omega_angles_in_radians(chain):

    phi = []
    psi = []
    omega = []
    chain.atom_to_internal_coordinates(verbose=False)
    for residue in chain:
        ric = residue.internal_coord
        if ric:
            phi_ang = ric.get_angle("phi")
            if phi_ang:
                phi.append(math.radians(phi_ang))
            else:
                phi.append(phi_ang)

            psi_ang = ric.get_angle("psi")
            if psi_ang:
                psi.append(math.radians(psi_ang))
            else:
                psi.append(psi_ang)

            omega_ang = ric.get_angle("omg")
            if omega_ang:
                omega.append(math.radians(omega_ang))
            else:
                omega.append(omega_ang)

    return (phi, psi, omega)
