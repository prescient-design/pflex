import os
import sys
import Bio
import pymol
import hydra
import subprocess
from Bio.PDB import *
from omegaconf import DictConfig

def split_pep_mhc_chains(pdbfile):

    pymol.cmd.load(pdbfile)
    pymol.cmd.do("select resi 181-")
    pymol.cmd.extract("pep", "sele")
    pymol.cmd.alter('pep and chain A', 'chain="B"')
    pymol.cmd.create('merged', "all")
    pymol.cmd.do("save "+pdbfile+", merged")
    pymol.cmd.do("delete all")

def renumber_residues(pdbfile, begin=1):
    """ Renumbers residues in a structure starting from begin (default: 1).
        Keeps numbering consistent with gaps if they exist in the original numbering.
    """

    p = PDBParser()
    structure = p.get_structure('X', pdbfile)
    for model in structure:
        for chain in model:
            if chain.get_list()[0].get_id()[1] == 0:
                return structure
            fresidue_num = chain.get_list()[0].get_id()[1]
            displace = begin - fresidue_num
            for res in chain:
                res.id = (res.id[0], res.id[1]+displace, res.id[2])

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdbfile)

def is_homolog_modeled(outdir):

    starting_model = ''
    for filename in os.listdir(outdir):
        if "_relaxed_0.pdb" in filename:
            starting_model = os.path.join(outdir, filename)
            break

    return starting_model

def homology_model_target_on_template(cfg:DictConfig):

    if not os.path.exists(cfg.outdir):
        os.makedirs(cfg.outdir, exist_ok=True)

    starting_model = is_homolog_modeled(cfg.outdir)
    
    if not (starting_model == ''):
        return starting_model

    os.system(f"cp {cfg.template_file} {cfg.outdir}")

    mhc_file = os.path.join(cfg.outdir, "mhc_list")
    pep_file = os.path.join(cfg.outdir, "pep_list")
    run_file = os.path.join(cfg.outdir, "run.sh")

    os.system(f"echo {cfg.allele} > {mhc_file}")
    os.system(f"echo \>{cfg.peptide}\'\n\'{cfg.peptide} > {pep_file}")

    exec_path = sys.executable
    template_pdb = os.path.join(cfg.outdir, cfg.template_pdb)

    if cfg.relax_after_threading:
        cmd = f"{exec_path} \
                {cfg.path_to_bin}/main.py \
                -nstruct {cfg.nstruct} \
                -relax_after_threading \
                -template_pdb {template_pdb} \
                -mhcs {mhc_file} \
                -peptides {pep_file} \
                -mhc_chain {cfg.mhc_chain} \
                -peptide_chain {cfg.peptide_chain} \
                -pep_start_index {cfg.pep_start_index} \
                -interface_cutpoint {cfg.interface_cutpoint} \
                -clustal_path {cfg.clustal_path}"
    else:
        cmd = f"{exec_path} \
                {cfg.path_to_bin}/mhc-pep-threader/main.py \
                -nstruct {cfg.nstruct} \
                -template_pdb {template_pdb} \
                -mhcs {mhc_file} \
                -peptides {pep_file} \
                -mhc_chain {cfg.mhc_chain} \
                -peptide_chain {cfg.peptide_chain} \
                -pep_start_index {cfg.pep_start_index} \
                -interface_cutpoint {cfg.interface_cutpoint} \
                -clustal_path {cfg.clustal_path}"

    os.system(f"echo {cmd} > {run_file}")
    os.system(f"chmod +x {run_file}")
    
    op = subprocess.run(run_file, shell=True, cwd=cfg.outdir)

    if op.returncode == 0:
        starting_model = is_homolog_modeled(cfg.outdir)
        split_pep_mhc_chains(starting_model)
        renumber_residues(starting_model)

    return starting_model