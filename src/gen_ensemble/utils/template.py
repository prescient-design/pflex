import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from Bio.PDB import PDBParser
from collections import defaultdict
from gen_ensemble.utils.dihedral_ops import *
from gen_ensemble.utils.mhc_pep_structure_ops import *
from gen_ensemble.constants.pseudomhcresidues import *
from Bio.SubsMat.MatrixInfo import *

BLOSUMX = defaultdict(dict)
for key in blosum62:
    key0 = key[0]
    key1 = key[1]
    BLOSUMX[key0][key1] = blosum62[key]
    BLOSUMX[key1][key0] = blosum62[key]
    BLOSUMX["-"][key0] = -50
    BLOSUMX[key0]["-"] = -50
    BLOSUMX[key1]["-"] = -50
    BLOSUMX["-"][key1] = -50
    BLOSUMX["-"]["-"] = -50


def get_seq_scores(seq1, seq2):

    total_score = 0
    for i in range(0, len(seq1)):

        total_score += BLOSUMX[seq1[i]][seq2[i]]

    return total_score

def random_template_for_each_pdb(cfg, pep_chain_map):

    template_for_each_pdb = defaultdict(list)
    pep_chains = list(pep_chain_map.keys())

    for i in range(0, len(pep_chains)):

        pdb1 = os.path.join(cfg.dir, pep_chains[i]+"_reordered.pdb")
        pep1 = read_structure_and_return_pep(pdb1)

        pdbs_to_choose_from = []

        if pep_chains[i] not in template_for_each_pdb:

            for j in range(i+1, len(pep_chains)):
                pdb2 = os.path.join(cfg.dir, pep_chains[j]+"_reordered.pdb")
                pep2 = read_structure_and_return_pep(pdb2)

                # works for the same length peptides
                if len(pep1) == len(pep2):
                    pdbs_to_choose_from.append(pdb2)

        if len(pdbs_to_choose_from) > 0:
            chosen = random.choice(pdbs_to_choose_from)
            template_for_each_pdb[pep_chains[i]] = [chosen]
            if chosen not in template_for_each_pdb:
                template_for_each_pdb[chosen] = [pep_chains[i]]
        else:
            template_for_each_pdb[pep_chains[i]] = []

    template_file = os.path.join(cfg.output.outdir, 'random_templates.csv')
    df = pd.DataFrame(template_for_each_pdb.items())
    df.to_csv(template_file,index=False)

    return template_for_each_pdb


def find_nearest_template_based_on_dist(distances_mean, distances_std, self_template, templates, mhc_chains, pep_chains, pep_lengths):

    pep_residues = [i for i in range(1,pep_lengths[self_template]+1)]

    avg_likelihood = {}

    for filename in os.listdir(templates):

        template_pdbid = filename.split("_")[0]

        if template_pdbid in pep_lengths and (pep_lengths[template_pdbid] == pep_lengths[self_template]) and ("-" not in fasta.pep_seq[template_pdbid]):
            if self_template not in filename:
                template_file = os.path.join(templates, filename)

                p = PDBParser(QUIET=True)
                s = p.get_structure('X', template_file)
                likelihood_est = []

                for model in s:
                    for i in range(0, len(PSEUDO_MHC_RES)):
                        for j in range(0, len(pep_residues)):

                            mhc_res_id = PSEUDO_MHC_RES[i]
                            pep_res_id = pep_residues[j]

                            ca_ca_distance = distances_mean[j][i]
                            ca_ca_sd = distances_std[j][i]

                            ca_ca_dist_in_temp = abs(model[mhc_chains[template_pdbid]][mhc_res_id]['CA'] - model[pep_chains[template_pdbid]][pep_res_id]['CA'])

                            prob_pdf = norm.logpdf(ca_ca_dist_in_temp, loc=ca_ca_distance, scale=ca_ca_sd)
                            likelihood_est.append(prob_pdf)
                            
                avg_likelihood[template_pdbid] = np.average(likelihood_est)

    sorted_likelihoods = {k: v for k, v in sorted(avg_likelihood.items(), key=lambda item: item[1], reverse=True)}
    first_item = next(iter(sorted_likelihoods))

    return [first_item]

def find_nearest_template_based_on_dist_dih(idx, vars, self_template, template_dir, fasta):

    pep_residues = [i for i in range(1, fasta.pep_lengths[self_template]+1)]

    avg_likelihood = {}
    for filename in os.listdir(template_dir):

        template_pdbid = filename.split("_")[0]

        if template_pdbid in fasta.pep_lengths and (fasta.pep_lengths[template_pdbid] == fasta.pep_lengths[self_template]) and ("-" not in fasta.pep_seq[template_pdbid]):
            if self_template not in filename:
                template_file = os.path.join(template_dir, filename)

                p = PDBParser(QUIET=True)
                s = p.get_structure('X', template_file)
                dist_likelihood_est = []
                phi_likelihood_est = []
                psi_likelihood_est = []

                for model in s:

                    (phi, psi, omega) = get_phi_psi_omega_angles(model[fasta.pep_chains[template_pdbid]])

                    for j in range(0, len(pep_residues)):

                        if phi[j]:
                            phi_prob_pdf = norm.logpdf(phi[j], loc=vars['phi_mean'][idx][j], scale=vars['phi_std'][idx][j])
                            phi_likelihood_est.append(phi_prob_pdf)

                        if psi[j]:
                            psi_prob_pdf = norm.logpdf(psi[j], loc=vars['psi_mean'][idx][j], scale=vars['psi_std'][idx][j])
                            psi_likelihood_est.append(psi_prob_pdf)


                    for i in range(0, len(PSEUDO_MHC_RES)):
                        for j in range(0, len(pep_residues)):

                            mhc_res_id = PSEUDO_MHC_RES[i]
                            pep_res_id = pep_residues[j]

                            ca_ca_distance = vars['distances_mean'][idx][j][i]
                            ca_ca_sd = vars['distances_std'][idx][j][i]

                            ca_ca_dist_in_temp = abs(model[fasta.mhc_chains[template_pdbid]][mhc_res_id]['CA'] - model[fasta.pep_chains[template_pdbid]][pep_res_id]['CA'])

                            prob_pdf = norm.logpdf(ca_ca_dist_in_temp, loc=ca_ca_distance, scale=ca_ca_sd)
                            dist_likelihood_est.append(prob_pdf)
                            
                avg_likelihood[template_pdbid] = np.average(dist_likelihood_est) + np.average(phi_likelihood_est) + np.average(psi_likelihood_est)

    sorted_likelihoods = {k: v for k, v in sorted(avg_likelihood.items(), key=lambda item: item[1], reverse=True)}
    first_item = next(iter(sorted_likelihoods))

    return [first_item]


def find_nearest_template_based_on_dist_dih_unk_seqs(idx, vars, target_peptide, template_dir, fasta):

    pep_residues = [i for i in range(1, len(target_peptide)+1)]

    avg_likelihood = {}
    for filename in os.listdir(template_dir):

        template_pdbid = filename.split("_")[0]

        if template_pdbid in fasta.pep_lengths and (fasta.pep_lengths[template_pdbid] == len(target_peptide)) and ("-" not in fasta.pep_seq[template_pdbid]):
            template_file = os.path.join(template_dir, filename)

            p = PDBParser(QUIET=True)
            s = p.get_structure('X', template_file)
            dist_likelihood_est = []
            phi_likelihood_est = []
            psi_likelihood_est = []

            for model in s:

                (phi, psi, omega) = get_phi_psi_omega_angles(model[fasta.pep_chains[template_pdbid]])

                for j in range(0, len(pep_residues)):

                    if phi[j]:
                        phi_prob_pdf = norm.logpdf(phi[j], loc=vars['phi_mean'][idx][j], scale=vars['phi_std'][idx][j])
                        phi_likelihood_est.append(phi_prob_pdf)

                    if psi[j]:
                        psi_prob_pdf = norm.logpdf(psi[j], loc=vars['psi_mean'][idx][j], scale=vars['psi_std'][idx][j])
                        psi_likelihood_est.append(psi_prob_pdf)


                for i in range(0, len(PSEUDO_MHC_RES)):
                    for j in range(0, len(pep_residues)):

                        mhc_res_id = PSEUDO_MHC_RES[i]
                        pep_res_id = pep_residues[j]

                        ca_ca_distance = vars['distances_mean'][idx][j][i]
                        ca_ca_sd = vars['distances_std'][idx][j][i]

                        ca_ca_dist_in_temp = abs(model[fasta.mhc_chains[template_pdbid]][mhc_res_id]['CA'] - model[fasta.pep_chains[template_pdbid]][pep_res_id]['CA'])

                        prob_pdf = norm.logpdf(ca_ca_dist_in_temp, loc=ca_ca_distance, scale=ca_ca_sd)
                        dist_likelihood_est.append(prob_pdf)
                        
            avg_likelihood[template_pdbid] = np.average(dist_likelihood_est) + np.average(phi_likelihood_est) + np.average(psi_likelihood_est)

    sorted_likelihoods = {k: v for k, v in sorted(avg_likelihood.items(), key=lambda item: item[1], reverse=True)}
    first_item = next(iter(sorted_likelihoods))

    return [first_item]

def closest_template_for_each_pdb_based_on_sequence(dir, pep_chain_map, pep_seq_map, pdbs_to_fold, benchmark=True):

    template_for_each_pdb = defaultdict(list)
    pep_chains = list(pep_chain_map.keys())

    if benchmark:
        for i in range(0, len(pep_chains)):

            if pep_chains[i] in pdbs_to_fold:

                pdb1 = os.path.join(dir, pep_chains[i]+"_reordered.pdb")
                pep1 = read_structure_and_return_pep(pdb1)
                pep_seq1 = pep_seq_map[pep_chains[i]]
                prev_score = -10000
                chosen = None

                if pep_chains[i] not in template_for_each_pdb:

                    for j in range(0, len(pep_chains)):
                        pdb2 = os.path.join(dir, pep_chains[j]+"_reordered.pdb")
                        pep2 = read_structure_and_return_pep(pdb2)
                        pep_seq2 = pep_seq_map[pep_chains[j]]

                        if len(pep_seq1) == len(pep_seq2) and pep_seq1 != pep_seq2 and "-" not in pep_seq2 and "-" not in pep_seq1:
                            cur_score = get_seq_scores(pep_seq1, pep_seq2)
                            if cur_score > prev_score:
                                prev_score = cur_score
                                chosen = pep_chains[j]

                    template_for_each_pdb[pep_chains[i]] = [chosen]
                    if chosen not in template_for_each_pdb:
                        template_for_each_pdb[chosen] = [pep_chains[i]]

                    print ("Chose template for ", pep_chains[i], chosen)
    else:
        for i in range(0, len(pdbs_to_fold)):
            pid = pdbs_to_fold[i]
            pep1 = pid.split("_")[0]

            prev_score = -10000
            chosen = None

            for pdbid in pep_seq_map:
                pep2 = pep_seq_map[pdbid]

                if len(pep1) == len(pep2) and ("-" not in pep2) and ("-" not in pep1):
                    cur_score = get_seq_scores(pep1, pep2)
                    if cur_score > prev_score:
                        prev_score = cur_score
                        chosen = pdbid

            template_for_each_pdb[pid] = [chosen]

    return template_for_each_pdb

def read_templates_from_multi_template_lib(dir, pep_len, fasta):

    target_dir = os.path.join(dir, str(pep_len))
    template_files = []
    mhc_chains = []
    pep_chains = []
    template = []

    try:
        for filename in os.listdir(target_dir):
            targetfile = os.path.join(target_dir, filename)
            template_files.append(targetfile)
            mhc_chains.append('A')
            pep_chains.append('B')
            template.append(filename)
    
    except Exception as e:
        print ("Multi template system is configured only for peptides of lengths 11, 12 and 13 AA")
        exit()

    return template_files, mhc_chains, pep_chains, template

