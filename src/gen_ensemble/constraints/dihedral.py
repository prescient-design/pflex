import math
from omegaconf import DictConfig
from collections import defaultdict

from pyrosetta import *
from pyrosetta.rosetta.core.scoring.func import CircularHarmonicFunc, CircularSplineFunc
from pyrosetta.rosetta.core.scoring.constraints import DihedralConstraint

from gen_ensemble.utils.pyrosetta_fns import *

class Dihedral:

    def __init__(self) -> None:
        pass

    def circularharmonic(self, mean: float, std: float):

        self.mean = math.radians(mean)
        self.std = math.radians(std)
        self.cfunc = CircularHarmonicFunc(x0_radians = self.mean, sd_radians = self.std)

    def circularspline(self, weight: float, prob_dist):

        self.prob_dist = prob_dist
        self.energy_dist = [-1 * math.log(x) for x in self.prob_dist]
        self.cfunc = CircularSplineFunc(weight_in = weight, energies_in = self.energy_dist)

def add_dihedral_constraints(pose: Pose, vars: defaultdict(list), idx: int, cfg: DictConfig, peptide: str, ctype: str) -> Pose:

    pep_residues = [i for i in range(1,len(peptide)+1)]

    # these chain ids are selected by default when homology modeling is performed
    mhc_chain = 'A'
    pep_chain = 'B'

    # PHI constraints
    for i in range(1, len(pep_residues)):

        dihedral = Dihedral()
        if ctype == "CIRCULAR_HARMONIC":
            
            phi_mean = vars['phi_mean'][idx][i]
            phi_sd = vars['phi_std'][idx][i]
            dihedral.circularharmonic(phi_mean, phi_sd)

        elif ctype == "CIRCULAR_SPLINE":

            phi_mean = vars['phi_mean'][idx][i]
            dihedral.circularspline(cfg.cst_fa_weight, phi_mean)

        atom1 = get_AtomID(pose, pep_chain, pep_residues[i-1], 'C')
        atom2 = get_AtomID(pose, pep_chain, pep_residues[i], 'N')
        atom3 = get_AtomID(pose, pep_chain, pep_residues[i], 'CA')
        atom4 = get_AtomID(pose, pep_chain, pep_residues[i], 'C')

        pose.add_constraint(DihedralConstraint(atom1, atom2, atom3, atom4, dihedral.cfunc))

    # PSI constraints

    for i in range(0, len(pep_residues)-1):

        dihedral = Dihedral()
        if ctype == "CIRCULAR_HARMONIC":

            psi_mean = vars['psi_mean'][idx][i]
            psi_sd = vars['psi_std'][idx][i]
            dihedral.circularharmonic(psi_mean, psi_sd)
        elif ctype == "CIRCULAR_SPLINE":

            psi_mean = vars['psi_mean'][idx][i]
            dihedral.circularspline(cfg.cst_fa_weight, psi_mean)

        atom1 = get_AtomID(pose, pep_chain, pep_residues[i], 'N')
        atom2 = get_AtomID(pose, pep_chain, pep_residues[i], 'CA')
        atom3 = get_AtomID(pose, pep_chain, pep_residues[i], 'C')
        atom4 = get_AtomID(pose, pep_chain, pep_residues[i+1], 'N')

        pose.add_constraint(DihedralConstraint(atom1, atom2, atom3, atom4, dihedral.cfunc))

    return pose

