
import os
from collections import defaultdict

from gen_ensemble.analyses.rms import *

RMSD_THRESHOLD = 0.5

def select_conserved_mhc_structures_from_MD_ensemble(mddir, nativedir, native_mhc_chains, native_pep_chains):

    pdb_md_map = defaultdict(list)

    for folder in os.listdir(mddir):

        targetfolder = os.path.join(mddir, folder)
        pdbid = folder
        for filename in os.listdir(targetfolder):

            if "no_na.cleaned.pdb" in filename:
                targetfile = os.path.join(targetfolder, filename)
                nativefile = os.path.join(nativedir, pdbid+"_reordered.pdb")

                target_mhc_chain = "A"
                native_mhc_chain = native_mhc_chains[pdbid]

                target_pep_chain = "B"
                native_pep_chain = native_pep_chains[pdbid]
                (rms_mhc, rms_pep) = get_bb_heavy_rmsd_for_mhcs(targetfile, nativefile, targetchain, nativechain)

                if rms_mhc <= RMSD_THRESHOLD:

                    pdb_md_map[pdbid].append(targetfile)



