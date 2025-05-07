
#project imports
import sys
import math
import itertools
import numpy as np
import pandas as pd

from collections import Counter

class GridGraph:

    def __init__(self, dataframe, D, chain, length, minpts_end, minpts_start, outdir):

        self.df = dataframe
        self.D = D
        self.chain = chain
        self.length = length
        self.minpts_end = minpts_end
        self.minpts_start = minpts_start
        self.FinalClusteringsDataFrame = pd.DataFrame(columns=['path', 'chain', 'clusterlabel'])
        self.FinalResultsDataFrame = pd.DataFrame(columns=['path', 'chain', 'finalcluster'])
        self.distance_cutoff = float(2.0)
        self.similarity_cutoff = float(1.00)
        self.structtoindexdict = dict()
        self.outdir = outdir

    def calculate_overlap(self, label1, label2):

        cluster1_df = self.FinalClusteringsDataFrame[self.FinalClusteringsDataFrame['clusterlabel']==label1]
        cluster2_df = self.FinalClusteringsDataFrame[self.FinalClusteringsDataFrame['clusterlabel']==label2]

        intersect = len([i for i in cluster1_df.index.values if i in cluster2_df.index.values])

        cluster1size = len(cluster1_df)
        cluster2size = len(cluster2_df)
        size_array = np.array([cluster1size, cluster2size])
        minsize = np.min(size_array)
        overlap_score = intersect / minsize

        if overlap_score >= self.similarity_cutoff:
            return 1

        if overlap_score < self.similarity_cutoff:
            return 0

    def connectgraph(self, graphdict, nodes, resultsdict):

        for node in nodes:

            subgraph = set([k for k, v in graphdict.items() if graphdict[node].intersection(v)])

            if subgraph:
                nodes = [i for i in nodes if i not in list(subgraph)]
                nodes = nodes
                resultsdict[node] = subgraph

                self.connectgraph(graphdict, nodes, resultsdict)

            else:
                continue

            if len(nodes) == 1 or len(nodes) == 0:
                return resultsdict
            break

    def get_clustering_labels(self):

        clustering_labels = list(set(self.postclustering_data['clusteringlabel'].values))
        return clustering_labels

    def get_clusters(self, clustering_label):

        clusters = list(set(self.postclustering_data['cluster'][self.postclustering_data['clusteringlabel']==clustering_label]))
        return clusters

    def run(self):

        if len(set(self.df['cluster'])) == 1 and list(set(self.df['cluster'].tolist()))[0] == -1:
            sys.exit('No clusters found over entire DBSCAN grid')

        df_nr = self.df[['path']].drop_duplicates().reset_index(drop=True)
        sample_df = self.df[self.df['clusteringlabel'] == 0].drop(['clusteringlabel', 'eps', 'minpts', 'cluster'], axis=1).reset_index(drop=True)
        sample_df['chain'] = self.chain
        for i in df_nr.index.values:

            if not df_nr.at[i, 'path'] in self.structtoindexdict:
                self.structtoindexdict[df_nr.at[i, 'path']] = dict()
                self.structtoindexdict[df_nr.at[i, 'path']] = i

            else:
                self.structtoindexdict[df_nr.at[i, 'path']] = i

        clusterlabel = 0

        for clusteringlabel in list(set(self.df['clusteringlabel'].tolist())):
            
            try:
                ClusteringDataFrame = self.df[self.df['clusteringlabel'] == clusteringlabel].reset_index(drop=True)
                clusters = list(set(ClusteringDataFrame['cluster'].tolist()))
                labels = ClusteringDataFrame['cluster'].tolist()
            except ValueError:
                print('no clusters found for clustering label ', clusterlabel)
                continue

            for cluster in clusters:
                if cluster != -1:

                    clusterindices = ClusteringDataFrame[ClusteringDataFrame['cluster'] == cluster].index.values
                    clusterdistances = self.D[np.ix_(clusterindices, clusterindices)]

                    if np.max(clusterdistances) <= self.distance_cutoff:
                        for i in ClusteringDataFrame[ClusteringDataFrame['cluster'] == cluster].index.values:

                            path = ClusteringDataFrame[ClusteringDataFrame['cluster'] == cluster].at[i, 'path']
                            temp_df = pd.DataFrame(columns=['path', 'chain', 'clusterlabel'])
                            temp_df.at[i, 'clusterlabel'] = clusterlabel
                            temp_df.at[i, 'path'] = path
                            temp_df.at[i, 'chain'] = self.chain
                            self.FinalClusteringsDataFrame = pd.concat([self.FinalClusteringsDataFrame, temp_df], axis=0)
                        clusterlabel += 1

        D_connection = np.zeros((len(list(set(self.FinalClusteringsDataFrame['clusterlabel']))), len(list(set(self.FinalClusteringsDataFrame['clusterlabel'])))))

        for cluster1, cluster2 in itertools.combinations(sorted(list(set(self.FinalClusteringsDataFrame['clusterlabel']))), 2):

            is_connected = self.calculate_overlap(cluster1, cluster2)
            D_connection[cluster1, cluster2] = is_connected
            D_connection[cluster2, cluster1] = is_connected

        graphdict = dict()
        nodes = range(len(D_connection))
        resultsdict = dict()

        for node in nodes:
            arr = D_connection[:][node]
            neighbors = list(np.where(arr == 1)[0])
            edges = neighbors + [node]
            graphdict[node] = set(edges)

        self.connectgraph(graphdict, nodes, resultsdict)

        finalclusterdict = dict()
        finalclusterlabel = 0

        for k, v in resultsdict.items():

            clusteritems = set()
            for node in v:
                clusterindices = set(self.FinalClusteringsDataFrame[self.FinalClusteringsDataFrame['clusterlabel'] == node].index.values)
                clusteritems = clusteritems.union(clusterindices)

            finalclusterdict[finalclusterlabel] = clusteritems
            finalclusterlabel = finalclusterlabel + 1

        for finalclusterlabel in finalclusterdict:
            for i in finalclusterdict[finalclusterlabel]:
                self.FinalResultsDataFrame.at[i, 'path'] = sample_df.at[i, 'path']
                self.FinalResultsDataFrame.at[i, 'chain'] = sample_df.at[i, 'chain']
                self.FinalResultsDataFrame.at[i, 'finalcluster'] = finalclusterlabel

        noisestructs = [v for v in sample_df.index.values if v not in self.FinalResultsDataFrame.index.values]

        for i in noisestructs:
            self.FinalResultsDataFrame.at[i, 'path'] = sample_df.at[i, 'path']
            self.FinalResultsDataFrame.at[i, 'chain'] = sample_df.at[i, 'chain']
            self.FinalResultsDataFrame.at[i, 'finalcluster'] = -1

        self.FinalResultsDataFrame = pd.merge(left=self.FinalResultsDataFrame, right=sample_df, left_on=['path', 'chain'], right_on=['path', 'chain'])

        lengthsdict = dict()

        for cluster in set(self.FinalResultsDataFrame.finalcluster.tolist()):
            lengthsdict[cluster] = len(self.FinalResultsDataFrame[self.FinalResultsDataFrame['finalcluster'] == cluster])

        passingclusters = [k for k, v in lengthsdict.items() if v > self.minpts_start]
        nonpassingindexes = [k for k in self.FinalResultsDataFrame.index[~self.FinalResultsDataFrame['finalcluster'].isin(passingclusters)]]

        for index in nonpassingindexes:
            self.FinalResultsDataFrame.at[index, 'finalcluster'] = -1

        self.FinalResultsDataFrame = self.FinalResultsDataFrame[self.FinalResultsDataFrame['finalcluster'].isin(passingclusters)]
        self.FinalResultsDataFrame.sort_values(by='finalcluster', ascending=False, inplace=True)

        clusters = list(set(self.FinalResultsDataFrame['finalcluster'].values))

        return self.FinalResultsDataFrame
