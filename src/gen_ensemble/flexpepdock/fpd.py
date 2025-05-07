#!/usr/bin/python

#     Prescient Design
#   Author: Santrupti Nerli
#   Date: Nov 16, 2023
#   Email: nerli.santrupti@gene.com
#

'''

FPD class contains all the necessary functionalities required to dock peptides
using FlexPepDock protocol.

'''

import hydra
from omegaconf import DictConfig
from collections import defaultdict

# import pyrosetta classes
from pyrosetta import *
from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingFlags, FlexPepDockingProtocol
from pyrosetta.toolbox.py_jobdistributor import PyJobDistributor
from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreType

from gen_ensemble.constraints.distance import *
from gen_ensemble.constraints.dihedral import *

class FPD:

    def __init__(self, cfg: DictConfig, vars: defaultdict(list), idx: int):

        self.cfg = cfg
        self.vars = vars
        self.idx = idx
        self.flags = FlexPepDockingFlags()
        self.sf = get_fa_scorefxn()
        self.init_flags()

    def init_flags(self):

        self.flags.set_peptide_chain(self.cfg.peptide_chain)
        self.flags.set_receptor_chain(self.cfg.receptor_chain)
        self.flags.pep_refine = self.cfg.pep_refine
        self.sf.set_weight(ScoreType.angle_constraint, self.cfg.cst_fa_weight)
        self.sf.set_weight(ScoreType.atom_pair_constraint, self.cfg.cst_fa_weight)
        self.sf.set_weight(ScoreType.dihedral_constraint, self.cfg.cst_fa_weight)
        self.intial_pose = pose_from_pdb(self.cfg.input_model)

    def distance_constraints(self, peptide):

        self.pose = add_flat_harmonic_distance_constraints(self.intial_pose, self.vars, self.idx, peptide)
    
    def dihedral_constraints(self, peptide, ctype):

        self.pose = add_dihedral_constraints(self.pose, self.vars, self.idx, self.cfg, peptide, ctype)

    # method to apply FlexPepDock mover to an input pose
    def fpd_apply(self) -> str:
        fpd = FlexPepDockingProtocol()
        filetag = self.cfg.input_model.replace(".pdb", self.cfg.suffix)
        jd = PyJobDistributor(filetag, self.cfg.nstruct, self.sf)

        while not jd.job_complete:
            pose = pose_from_sequence("AAAAA") 
            pose.assign(self.pose)
            fpd.apply(pose)
            jd.output_decoy(pose)

        return filetag
        