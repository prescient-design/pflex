
from collections import defaultdict

from pyrosetta import *
from pyrosetta.rosetta.core.scoring.func import FlatHarmonicFunc
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint

from gen_ensemble.constants.pseudomhcresidues import *
from gen_ensemble.utils.pyrosetta_fns import *

class Distance:

    def __init__(self) -> None:
        pass

    def flatharmonicfunc(self, mean: float, std: float, tol: float):

        self.mean = mean
        self.std = std
        self.tol = tol
        self.harm_func = FlatHarmonicFunc(x0_in = self.mean, sd_in = self.std, tol_in = self.tol)


def add_flat_harmonic_distance_constraints(pose: Pose, vars: defaultdict(list), idx: int, peptide: str) -> Pose:

    pep_residues = [i for i in range(1,len(peptide)+1)]

    # these chain ids are selected by default when homology modeling is performed
    mhc_chain = 'A'
    pep_chain = 'B'

    for i in range(0, len(PSEUDO_MHC_RES)):
        for j in range(0, len(pep_residues)):

            mean = vars['distances_mean'][idx][j][i]
            sd = vars['distances_std'][idx][j][i]
            tol = round(3.0*sd, 2)
            distance = Distance()
            distance.flatharmonicfunc(mean, sd, tol)

            atom1 = get_AtomID(pose, mhc_chain, PSEUDO_MHC_RES[i], 'CA')
            atom2 = get_AtomID(pose, pep_chain, pep_residues[j], 'CA')

            pose.add_constraint(AtomPairConstraint(atom1, atom2, distance.harm_func))

    return pose


