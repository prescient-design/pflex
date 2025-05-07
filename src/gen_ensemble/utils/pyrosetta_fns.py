
from pyrosetta import *

def get_AtomID(pose: Pose, chain: str, resi: int, atomname: str) -> AtomID:

    resi = pose.pdb_info().pdb2pose(res=resi, chain=chain)
    residue = pose.residue(resi)
    return AtomID(atomno_in = residue.atom_index(atomname), rsd_in = resi)