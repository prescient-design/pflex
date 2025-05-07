import os
import ast
import sys
import json
import subprocess

from gen_ensemble.analyses.rms import *
from gen_ensemble.constants.constants import *
from gen_ensemble.analyses.contact_metric import *
from gen_ensemble.analyses.dihedral_distance import *
from gen_ensemble.utils.mhc_pep_structure_ops import *

def read_scores(rundir, filetag):
        
    score_file = os.path.join(rundir, filetag+".fasc")
    scores = {}
    with open(score_file, "r") as inputfilehandler:
        for line in inputfilehandler:
            data = ast.literal_eval(line)
            tag = os.path.basename(data['decoy']).replace(".pdb", "")
            scores[tag] = data['total_score']

    return scores

def compare_docked_with_native(rundir, nativedir, filetag, native_pdbid):

    scores = read_scores(rundir, filetag)

    outfile = os.path.join(rundir, native_pdbid+".csv")
    outfilehandler = open(outfile, 'w')

    scorefile = os.path.join(rundir, native_pdbid+"_scores.csv")
    scorefilehandler = open(scorefile, 'w')
    scorefilehandler.write("tag,score,avg_all,avg_phi,avg_psi,avg_omega,max_all,max_phi,max_psi,max_omega,rms,med_all,med_phi,med_psi,med_omega,cm\n")

    native = os.path.join(nativedir, native_pdbid+"_reordered.pdb")
    native_pep = read_structure_and_return_pep(native)

    #ignore_list = identify_rigid_pep_residues(native)
    ignore_list = []

    max_distances = []
    avg_distances = []
    med_distances = []
    max_foi = ''
    avg_foi = ''
    med_foi = ''
    prev_max_all = MAX_DIH_DIST
    prev_avg_all = MAX_DIH_DIST
    prev_med_all = MAX_DIH_DIST


    for filename in os.listdir(rundir):
        if "dock_refined_" in filename and ".pdb" in filename:
            targetfile = os.path.join(rundir, filename)
            target_pep = read_structure_and_return_pep(targetfile)

            (max_phi, max_psi, max_all, max_omega) = max_distance(native_pep, target_pep, ignore_list)
            (avg_phi, avg_psi, avg_all, avg_omega) = average_distance(native_pep, target_pep, ignore_list)
            (med_phi, med_psi, med_all, med_omega) = median_distance(native_pep, target_pep, ignore_list)

            rms = get_bb_heavy_rmsd(native, targetfile, native_pep.id, target_pep.id)

            xtal = PDB(native)
            model = PDB(targetfile)

            cm = contact_metric(model, xtal, threshold=3.1, xtal_peptide_chain=native_pep.id, model_peptide_chain=target_pep.id)

            tag = filename.replace(".pdb", '')
            if tag in scores:
                score = scores[tag]
                scorefilehandler.write(tag+","+str(score)+","+str(avg_all)+","+str(avg_phi)+","+str(avg_psi)+","+str(avg_omega)+","+str(max_all)+","+str(max_phi)+","+str(max_psi)+","+str(max_omega)+","+str(rms)+","+str(med_all)+","+str(med_phi)+","+str(med_psi)+","+str(med_omega)+","+str(cm)+"\n")

            if max_psi != None:
                max_distances.append(max_all)
                if max_all < prev_max_all:
                    prev_max_all = max_all
                    max_foi = filename.replace(".pdb", '')

            if avg_psi != None:
                avg_distances.append(avg_all)
                if avg_all < prev_avg_all:
                    prev_avg_all = avg_all
                    avg_foi = filename.replace(".pdb", '')

            if med_psi != None:
                med_distances.append(avg_all)
                if med_all < prev_med_all:
                    prev_med_all = med_all
                    med_foi = filename.replace(".pdb", '')

    if len(max_distances) > 0 and len(avg_distances) > 0: 
        outfilehandler.write(native_pdbid+",a,"+str(min(avg_distances))+","+str(prev_avg_all)+","+avg_foi+","+str(len(avg_distances))+","+str(min(max_distances))+","+str(prev_max_all)+","+max_foi+","+str(len(max_distances))+",full\n") 

    outfilehandler.close()
    scorefilehandler.close()
    