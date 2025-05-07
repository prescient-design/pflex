
#math imports
import numpy as np
import math


#Project Imports
import itertools
import pandas as pd
from sklearn.cluster import dbscan
from collections import defaultdict, Counter

class DBSCAN:

    """Cluster peptide conformations using cosine difference measurements for each pdb path"""

    def __init__(self, D_pairwise, preclustering_data, pep_length, eps_start=0.2, eps_end=1.2, eps_step=0.2, minpts_start=2, minpts_end=15, minpts_step=1):

        self.D_pairwise = D_pairwise
        self.preclustering_data = preclustering_data
        self.postclustering_data = pd.DataFrame(columns=['clusteringlabel', 'cluster', 'eps', 'minpts', 'path'])
        self.eps = 0.0
        self.MinPts = 0
        self.eps_vals = np.arange(eps_start, eps_end, eps_step)
        self.MinPts_vals = np.arange(minpts_start, minpts_end, minpts_step)
        self.clustering = ''
        self.pep_length = pep_length

    def set_eps(self, eps):

        self.eps=eps
        return

    def get_eps(self):
        return self.eps

    def set_MinPts(self, MinPts):

        self.MinPts = MinPts

        return

    def get_MinPts(self):
        return self.MinPts

    def run_dbscan(self):

        cluster_label = 0
        for params in itertools.product(self.eps_vals, self.MinPts_vals):

            temp_df = pd.DataFrame(columns=self.postclustering_data.columns)

            eps = params[0]
            MinPts = params[1]

            print('running DBSCAN for eps '+str(eps)+' and MinPts '+str(MinPts))

            self.set_eps(eps)
            self.set_MinPts(MinPts)

            self.clustering = dbscan(self.D_pairwise, eps=self.eps, min_samples=self.MinPts, metric='precomputed')
            self.save_clustering_info(cluster_label, temp_df)

            cluster_label += 1

        return self.postclustering_data

    def convert_to_degrees(self):

        for variable in ['phi', 'psi', 'omega']:
            self.preclustering_data[variable] = self.preclustering_data[variable].apply(math.degrees)

        return self.preclustering_data

    def save_clustering_info(self, cluster_label, temp_df):

        temp_df['clusteringlabel'] = [cluster_label] * len(self.D_pairwise)
        
        clusters = self.clustering[1]
        temp_df['cluster'] = clusters

        temp_df['eps'] = [self.eps] * len(self.D_pairwise)
        temp_df['minpts'] = [self.MinPts] * len(self.D_pairwise)

        indexes = self.preclustering_data.index.values

        for i in indexes:
            phi_psi_vals = []

        temp_df['path'] = sorted(list(set(self.preclustering_data['path'].tolist())))
        self.postclustering_data = pd.concat([self.postclustering_data, temp_df], axis=0)

        return
