import pandas as pd
import numpy as np
import os


def load_pdb_data(version):
    # df = pd.read_parquet('/gstore/home/parj2/stage/tcr_design/philia/notebooks/'
    #                      'formatted_train_set.parquet')
    df = pd.read_parquet(f's3://prescient-data-dev/sandbox/luc42/tcr_pmhc/reformatted_train_set_v{version}.parquet')
    return df

def load_pdb_data_log(version):
    # df = pd.read_parquet('/gstore/home/parj2/stage/tcr_design/philia/notebooks/'
    #                      'formatted_train_set.parquet')
    df = pd.read_parquet(f's3://prescient-data-dev/sandbox/luc42/tcr_pmhc/reformatted_train_set_v{version}_log_dist.parquet')
    return df
# def load_val_data(version):
#     if version == 4:
#         data_path = 's3://prescient-data-dev/sandbox/nerlis/scratch/pmhc_db_splits_feb7/splits/'
        
# def load_pdb_data_log_flip(version):
#     df = pd.read_parquet(f's3://prescient-data-dev/sandbox/luc42/tcr_pmhc/reformatted_train_set_v{version}_log_dist_flip_angles.parquet')
#     return df
    
def load_transphla_data():
    df = pd.read_csv('s3://prescient-data-dev/sandbox/parj2/tcr_pmhc/sequence_train_set.csv',
                     index_col=None).reset_index(drop=True)
    return df


def get_group_index(indices, group_ratio, seed=42):
    n_data = len(indices)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(indices)
    n_groups = len(group_ratio)
    to_slice = np.cumsum(np.array(group_ratio)*n_data).astype(int)
    return np.split(indices, to_slice[:n_groups])[:n_groups]


def read_from_file(filename):
    with open(filename, "r") as inpfilehandler:
        contents = inpfilehandler.read()
    return contents


def load_pdb_split(path):
    indices = read_from_file(path)
    indices_list = indices[:-1].split(',')  # [:-1] removes trailing \n
    return indices_list

def load_pdb_split_bulk(children_path, parent_path, val_pdb_only=False):
    # if str(children_path)[0] == '[':
    version, cross_val = children_path.split('-')
    if int(version) == 4:
        candidates = ['noisy', 'augmented_set2']
    elif int(version) == 3:
        candidates = ['noisy', 'augmented_set1']
    elif int(version) == 2:
        candidates = ['noisy']
    elif int(version) == 1:
        candidates = ['clean']
    elif int(version) == 8:
        candidates = ['noisy', 'augmented_set2']
    elif int(version) == 9:
        candidates = ['augmented']
    if val_pdb_only:
        if int(version) == 1:
            candidates = ['clean']
        else:
            candidates = ['noisy']
    print(f'candidates are from: {candidates}')
    bulk_list = []
    for cand in candidates:
        path = os.path.join(parent_path, os.path.join(cand, cross_val))
        child_list = load_pdb_split(path)
        bulk_list += child_list
    return bulk_list
    # else:
        # return load_pdb_split(os.path.join(parent_path, children_path))

def get_df_indices(df, pdb_ids):
    df = df.reset_index(drop=True)
    return df[df['pdbid'].isin(pdb_ids)].index.values


def get_indices_from_file(path, df, index_type):
    if index_type == 'pdbid':
        pdb_ids_list = load_pdb_split(path)
        return get_df_indices(df, pdb_ids_list)
    elif index_type == 'row':
        # TODO: support other loading schemes
        row_ids_list = list(np.loadtxt(path).astype(int))
        return row_ids_list
    else:
        raise NotImplementedError

def add_relax(pdbids, num, remove_raw=False):
    # num: number of relax structures for the corresponding af2 structures
    relaxed_pdbids = []
    af2_pdbids = [pdbid for pdbid in pdbids if len(pdbid) > 4]
    for pdbid in af2_pdbids:
        for n in range(num):
            relaxed_pdbids.append(pdbid + '_000' + str(n+1))
    if remove_raw:
        return relaxed_pdbids
    return pdbids + relaxed_pdbids