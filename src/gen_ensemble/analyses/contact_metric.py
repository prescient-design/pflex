import os
import ast
import sys
import json
import subprocess

class PDB:

    def __init__(self, pdb_filename):

        self.chains = {}
        def chain_of(line): return line[21]
        def residue_of(line): return line[22:26]
        def atom_of(line): return line[12:16]
        def xyz_of(line): return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        def elem_of(line): return line[76:].strip()

        with open(pdb_filename, 'r') as f:
            for line in f:
                if line[:6] not in {'ATOM  ', 'HETATM'} : continue

                # skip all hydrogens
                if elem_of(line) == 'H': continue

                chain = chain_of(line)
                residue = residue_of(line)
                atom = atom_of(line)
                xyz = xyz_of(line)
                if chain in self.chains:
                    if residue in self.chains[chain]: 
                        self.chains[chain][residue][atom] = xyz
                    else:
                        self.chains[chain][residue] = {atom: xyz}
                else:
                    self.chains[chain] = {residue: {atom: xyz}}

def dist_sq(xyz1, xyz2):
    return (xyz1[0] - xyz2[0])**2 + (xyz1[1] - xyz2[1])**2 + (xyz1[2] - xyz2[2])**2

def count_contacts(pdb, threshold, peptide_chain):
    contacts = []
    # compare dist sq to thresh sq (faster)
    thresh2 = threshold * threshold

    # first: what are the residues on non peptide whose CAs are within 15? skip all else
    # for chain_letter, contents in pdb.chains.items():
    peptide_CA_xyzs = [
        pdb.chains[peptide_chain][r][' CA '] for r in pdb.chains[peptide_chain].keys()
    ]

    mhc_chain = [c for c in pdb.chains.keys() if c != peptide_chain]
    assert len(mhc_chain) == 1
    mhc_chain = mhc_chain[0]

    nearby_mhc_residues = [
        r for r in pdb.chains[mhc_chain].keys() if min([
            dist_sq(pdb.chains[mhc_chain][r][' CA '], p) for p in peptide_CA_xyzs
        ]) <= 144
    ]

    for peptide_residue in pdb.chains[peptide_chain].keys():
        for mhc_residue in nearby_mhc_residues:
            # skip this nearby mhc residue, before even evaluating an atom pair, if its
            # CA distance is further than something more stringent. Maybe...

            # are there any atom-atom contacts?
            for peptide_atom, paxyz in pdb.chains[peptide_chain][peptide_residue].items():
                for mhc_atom, maxyz in pdb.chains[mhc_chain][mhc_residue].items():
                    if dist_sq(paxyz, maxyz) <= thresh2:
                        contacts.append((peptide_residue, peptide_atom, mhc_residue, mhc_atom))
    
    return contacts

def contact_metric(model, xtal, threshold, xtal_peptide_chain, model_peptide_chain):
    """
    Counts all the contacts between chains in xtal, then sees what fraction are
    recovered in model. Contacts are distances below threshold.
    """

    model_contacts = set(count_contacts(model, threshold, model_peptide_chain))
    xtal_contacts = set(count_contacts(xtal, threshold, xtal_peptide_chain))

    # we sort of have multiple contact metrics. we have an f1 score we could compute!
    # but for the moment, let's just think of how many contacts we correctly recover,
    # our TPR.
    return len([mc for mc in model_contacts if mc in xtal_contacts]) / len(xtal_contacts)