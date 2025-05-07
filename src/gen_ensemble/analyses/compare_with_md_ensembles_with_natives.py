
import os
import pandas as pd
from omegaconf import DictConfig
from collections import defaultdict

from gen_ensemble.analyses.rms import *
from gen_ensemble.utils.file_ops import *
from gen_ensemble.analyses.contact_metric import *
from gen_ensemble.analyses.dihedral_distance import *
from gen_ensemble.utils.mhc_pep_structure_ops import *

RMSD_THRESHOLD = 20.0

def select_conserved_mhc_structures_from_MD_ensemble_and_measure_rms(cfg: DictConfig, native_mhc_chains, native_pep_chains):

    md_map_df = pd.DataFrame(columns=['pdbid', 'md_sample', 'mhc_rms_to_native', 'pep_rms_to_native', 'dscore', 'dscore_no_omega', 'cm', 'avg_dih_dist', 'max_dih_dist', 'med_dih_dist'])

    if not os.path.exists(cfg.md.rms_to_native):
        os.system(f"mkdir {cfg.md.rms_to_native}")

    target_mhc_chain = "A"
    target_pep_chain = "B"

    for folder in os.listdir(cfg.md.dir):

        targetfolder = os.path.join(cfg.md.dir, folder)
        pdbid = folder
        if pdbid in native_mhc_chains and pdbid in native_pep_chains:
            
            nativefile = os.path.join(cfg.dir, pdbid+"_reordered.pdb")
            native_mhc_chain = native_mhc_chains[pdbid]
            native_pep_chain = native_pep_chains[pdbid]

            if os.path.exists(nativefile):
                for filename in os.listdir(targetfolder):

                    if "no_na.cleaned.pdb" in filename:
                        
                        targetfile = os.path.join(targetfolder, filename)
                        computed = ((md_map_df['md_sample'] == targetfile)).any()

                        if not computed:                    
                            (rms_mhc, rms_pep) = get_bb_heavy_rmsd_for_overlapping_mhcs(targetfile, nativefile, target_pep_chain, native_pep_chain, target_mhc_chain, native_mhc_chain)

                            if rms_mhc <= RMSD_THRESHOLD:
                                
                                target_pep = read_structure_and_return_pep(targetfile)
                                native_pep = read_structure_and_return_pep(nativefile)

                                avg_dih_dist = average_distance(target_pep, native_pep)
                                max_dih_dist = max_distance(target_pep, native_pep)
                                med_dih_dist = median_distance(target_pep, native_pep)

                                dscore = d_score(targetfile, nativefile, target_pep_chain, native_pep_chains[pdbid], pep_residues, pep_residues)
                                dscore_no_omega = d_score_no_omega(targetfile, nativefile, target_pep_chain, native_pep_chains[pdbid], pep_residues, pep_residues)

                                native_sample = PDB(nativefile)
                                model_sample = PDB(targetfile)

                                cm = contact_metric(model_sample, native_sample, threshold=3.1, xtal_peptide_chain=native_pep.id, model_peptide_chain=target_pep.id)

                                md_map_df.loc[len(md_map_df.index)] = [pdbid, targetfile, rms_mhc, rms_pep, dscore, dscore_no_omega, cm, avg_dih_dist, max_dih_dist, med_dih_dist] 

    md_map_df.to_csv(os.path.join(cfg.md.rms_to_native, "md_rms_to_native.tsv"), sep="\t", index=False)

def replace_path(p, pdbid, md_dir):

    return os.path.join(md_dir, pdbid, os.path.basename(p))

def compare_MD_and_docked_ensembles(cfg: DictConfig, pep_len):

    md_samples_df = pd.read_csv(os.path.join(cfg.md.rms_to_native, "md_rms_to_native.tsv"), sep="\t")

    mhc_chain = "A"
    pep_chain = "B"

    pep_residues = [i for i in range(1,pep_len+1)]

    for folder in os.listdir(cfg.output.outdir):

        targetfolder = os.path.join(cfg.output.outdir, folder)
        pdbid = folder
        pdbid_md_samples = md_samples_df.loc[md_samples_df['pdbid'] == pdbid]

        print (pdbid, cfg.output.outdir)

        pdbid_md_samples['md_sample'] = pdbid_md_samples['md_sample'].apply(replace_path, args=(pdbid, cfg.md.dir))
        md_samples = pdbid_md_samples['md_sample'].tolist()
        
        if os.path.isdir(targetfolder) and "hydra" not in folder:
            
            for subfolder in os.listdir(targetfolder):
                targetsubfolder = os.path.join(targetfolder, subfolder)
                md_model_map_df = pd.DataFrame(columns=['pdbid', 'md_sample', 'model', 'mhc_rms', 'pep_rms', 'avg_dih_dist', 'max_dih_dist', 'dscore', 'dscore_no_omega'])

                for filename in os.listdir(targetsubfolder):
                    if "_dock_refined" in filename and ".pdb" in filename:

                        targetfile = os.path.join(targetsubfolder, filename)
                        model_pep = read_structure_and_return_pep(targetfile)
                        
                        for i in range(0, len(md_samples)):
                            (rms_mhc, rms_pep) = get_bb_heavy_rmsd_for_overlapping_mhcs(targetfile, md_samples[i], pep_chain, pep_chain, mhc_chain, mhc_chain)
                            md_pep = read_structure_and_return_pep(md_samples[i])

                            avg_dih_dist = average_distance(model_pep, md_pep)
                            max_dih_dist = max_distance(model_pep, md_pep)

                            dscore = d_score(targetfile, md_samples[i], pep_chain, pep_chain, pep_residues, pep_residues)
                            dscore_no_omega = d_score_no_omega(targetfile, md_samples[i], pep_chain, pep_chain, pep_residues, pep_residues)

                            md_model_map_df.loc[len(md_model_map_df.index)] = [pdbid, md_samples[i], targetfile, rms_mhc, rms_pep, avg_dih_dist, max_dih_dist, dscore, dscore_no_omega]

                md_model_map_df.to_csv(os.path.join(targetsubfolder, "md_model_comparison.tsv"), sep="\t", index=False)
                        

