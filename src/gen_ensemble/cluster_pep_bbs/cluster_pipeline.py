#Python imports
import os
import timeit
import pyrosetta
import itertools
import numpy as np
import pandas as pd

from gen_ensemble.utils.dihedral_ops import *
from gen_ensemble.cluster_pep_bbs.utils import *
from gen_ensemble.cluster_pep_bbs.DBSCAN import *
from gen_ensemble.cluster_pep_bbs.GridGraph import *
from gen_ensemble.utils.mhc_pep_structure_ops import *

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from collections import defaultdict
from sklearn.metrics import pairwise_distances_argmin_min

def run_cluster_pipeline(rundir, pep_chain, cfg):

    # create project dir
    if not os.path.exists(rundir):
        os.system('mkdir ' + rundir)

    i = 0
    starting_res = 1
    pep_length = 0

    pep_df = pd.DataFrame(columns=['path', 'chain', 'length', 'residue', 'phi', 'psi', 'omega'])
    for filename in os.listdir(rundir):

        if ("dock_refined_" in filename and ".pdb" in filename) or ("no_na.cleaned.pdb" in filename):
            targetfile = os.path.join(rundir, filename)

            target_pep = read_structure_and_return_pep(targetfile)
            (phi, psi, omega) = get_phi_psi_omega_angles_in_radians(target_pep)

            residues = [i for i in range(starting_res, len(target_pep))]
            pathlist = [targetfile for i in range(starting_res, len(target_pep))]
            chainlist = [pep_chain for i in range(starting_res, len(target_pep))]
            lengthlist = [len(target_pep) for i in range(starting_res, len(target_pep))]
            philist = [phi[i] for i in range(starting_res, len(target_pep))]
            psilist = [psi[i] for i in range(starting_res, len(target_pep))]
            omegalist = [omega[i] for i in range(starting_res, len(target_pep))]

            row = pd.DataFrame.from_dict({'path': pathlist, 'chain': chainlist, 'length': lengthlist, 'residue': residues, 'phi': philist, 'psi': psilist, 'omega': omegalist})
            pep_df = pd.concat([pep_df, row])

            if pep_length == 0:
                pep_length = len(target_pep)


    pep_df = pep_df.sort_values(by=['path', 'residue'])
    pep_df['id'] = pep_df.groupby(['path']).ngroup()
    pep_df = pep_df.set_index('id')

    PHI = np.zeros((len(set(pep_df.index.values)), pep_length-starting_res), dtype=np.float64)
    PSI = np.zeros((len(set(pep_df.index.values)), pep_length-starting_res), dtype=np.float64)
    OMEGA = np.zeros((len(set(pep_df.index.values)), pep_length-starting_res), dtype=np.float64)
        
    for index in set(pep_df.index.values):

        phiarr = np.array(pep_df['phi'][pep_df.index==index].tolist())
        psiarr = np.array(pep_df['psi'][pep_df.index==index].tolist())
        omegaarr = np.array(pep_df['omega'][pep_df.index==index].tolist())

        PHI[index] = phiarr
        PSI[index] = psiarr
        OMEGA[index] = omegaarr

    D_pairwise = calculate_cosine_difference(PHI, PSI, OMEGA, len(set(pep_df.index.values)))
    print(D_pairwise)

    if cfg.cluster.method == "dbscan":
        dbscan = DBSCAN(D_pairwise=D_pairwise, preclustering_data=pep_df, pep_length=pep_length, eps_start=0.2, eps_end=1.2, eps_step=0.2, minpts_start=2, minpts_end=15, minpts_step=1)
        clustering_dataframe = dbscan.run_dbscan()
        clustering_dataframe.to_csv('test.csv')

        gridgraph = GridGraph(dataframe=clustering_dataframe, D=D_pairwise, chain=pep_chain, length=pep_length, minpts_end=15, minpts_start=2, outdir=rundir)
        final_dataframe = gridgraph.run()
        
        final_dataframe.to_csv(os.path.join(rundir,'clustering_results.csv'))
    elif cfg.cluster.method == 'agglomerative':

        k = cfg.cluster.k_start
        while k < cfg.cluster.k_end:

            model = AgglomerativeClustering(affinity='precomputed', n_clusters = None, distance_threshold = k, linkage='complete').fit(D_pairwise)
            y_predict = AgglomerativeClustering(affinity='precomputed', n_clusters = None, distance_threshold = k, linkage='complete').fit_predict(D_pairwise)

            clusters = defaultdict(list)
            for i in range(0, len(model.labels_)):
                clusters[model.labels_[i]].append(i)

            print (k, len(clusters))

            clf = NearestCentroid()
            clf.fit(D_pairwise, y_predict)
            closest, _ = pairwise_distances_argmin_min(clf.centroids_, D_pairwise)
            grouped = pep_df.drop_duplicates(subset=['path'], keep='last')

            with open(os.path.join(rundir, "clustering_results_"+str(k)+".csv"), 'w') as outfilehandler:
                for c in closest:
                    outfilehandler.write(str(c)+","+grouped.iloc[c]['path']+"\n")
            
            k += cfg.cluster.k_step
        



