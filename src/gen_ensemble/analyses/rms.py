
import pymol

def get_bb_heavy_rmsd(pdb1, pdb2, chain_pdb1, chain_pdb2):
    pymol.cmd.load(pdb1, "obj01")
    pymol.cmd.load(pdb2, "obj02")

    chain1 = "obj01 and chain "+chain_pdb1+" and name n+c+ca+o"
    chain2 = "obj02 and chain "+chain_pdb2+" and name n+c+ca+o"

    pymol.cmd.super("obj01", "obj02", cycles=0)
    rms = pymol.cmd.rms_cur(chain1, chain2, cycles=0, matchmaker=4)
    pymol.cmd.do("delete all")

    return rms

def get_bb_heavy_rmsd_for_overlapping_mhcs(pdb1, pdb2, chain_pdb1, chain_pdb2, mhc_chain1, mhc_chain2):
    pymol.cmd.load(pdb1, "obj01")
    pymol.cmd.load(pdb2, "obj02")

    chain1 = "obj01 and chain "+chain_pdb1+" and name n+c+ca+o"
    chain2 = "obj02 and chain "+chain_pdb2+" and name n+c+ca+o"

    mchain1 = "obj01 and chain "+mhc_chain1+" and name n+c+ca+o"
    mchain2 = "obj02 and chain "+mhc_chain2+" and name n+c+ca+o"

    pymol.cmd.super("obj01", "obj02", cycles=0)

    rms_pep = pymol.cmd.rms_cur(chain1, chain2, cycles=0, matchmaker=4)
    rms_mhc = pymol.cmd.rms_cur(mchain1, mchain2, cycles=0, matchmaker=4)
    pymol.cmd.do("delete all")

    return (rms_mhc, rms_pep)
