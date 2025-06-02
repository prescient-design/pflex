import sys
import pandas as pd
import numpy as np
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from Bio import SeqIO

from philia.trainval_data.transforms import (
    scale_dist, scale_dist_log, scale_angles_separately, scale_angles)#, scale_angles_flip)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def format_dist(x):
    x = np.stack(x, axis=0)
    return x


def format_angles(x):
    x = np.stack(x, axis=0)  # [L, 2] where L = peptide length
    # x[0, 0] = x[-1, 0]  # bring second-to-last to first nan slot
    # x = x[:-1]  # [L-1, 2]
    return x

def pad(x):
    # x ~ [L, 2] with variable L that is max 15
    L, _ = x.shape
    return np.pad(x, [(0, 15-L), (0, 0)], constant_values=np.nan)  # y ~ [15, 2]

def load_pdb(databases):
    if str(databases)[0] == '[':
        df_pdb = load_pdb_bulk(databases)
    else:
        df_pdb = load_pdb_single(databases)
    return df_pdb

def load_pdb_single(db):
    return pd.read_parquet(db)

def load_pdb_bulk(dbs):
    df_pdb = load_pdb_single(dbs[0])
    for i in range(1,len(dbs)):
        df_pdb_augmented = load_pdb_single(dbs[i])
        df_pdb = pd.concat([df_pdb, df_pdb_augmented], ignore_index=True)
    df_pdb = df_pdb[~df_pdb['pdbid'].isnull()]
    df_pdb = df_pdb[['pdbid', 'allele', 'peptide', 'pseudo_mhc', 'dihedral_matrix', 'ca_distance_matrix', 'cb_distance_matrix']]
    df_pdb.rename(columns={'pseudo_mhc': 'hla_sequence',
                    'allele': 'hla',
                    'ca_distance_matrix': 'distance_matrix'}, inplace=True)
    return df_pdb

def load_seq_pdb(peptide_set, hla_set, maps):
    # Read in peptide seqs
    n_peptides = 0
    pep_df = pd.DataFrame()
    for record in SeqIO.parse(peptide_set, "fasta"):
        n_peptides += 1
        meta_split = record.id.split('_')
        allele_info = f'{meta_split[1][0]}*{meta_split[1][1:3]}:{meta_split[1][-2:]}'
        entry = pd.DataFrame.from_dict({"peptide": [''.join(record.seq)],
                                        "pdbid": [meta_split[0]],
                                        'hla': [allele_info]})
        pep_df = pd.concat([pep_df, entry], ignore_index=True)
    logger.info(f'Number of peptides: {n_peptides}')
    # Read in HLA seqs
    n_hla = 0
    hla_df = pd.DataFrame()
    for record in SeqIO.parse(hla_set, "fasta"):
        n_hla += 1
        meta_split = record.id.split('_')
        allele_info = f'{meta_split[1][0]}*{meta_split[1][1:3]}:{meta_split[1][-2:]}'
        entry = pd.DataFrame.from_dict({"hla_sequence": [''.join(record.seq)],
                                        "pdbid": [meta_split[0]],
                                        'hla': [allele_info]})
        hla_df = pd.concat([hla_df, entry], ignore_index=True)
    logger.info(f'Number of HLA: {n_hla}')

    # Merge peptide, HLA seqs first
    df_seqs = pep_df.merge(hla_df, how='inner')
    df_labels = pd.read_parquet(maps)
    df_pdb = df_labels.merge(df_seqs, how='inner').reset_index(drop=True)
    return df_pdb

@hydra.main(version_base=None, config_path="dconf", config_name="dconfig")
def main(cfg: DictConfig) -> None:
    if not cfg.extra_seq_import:
        df_pdb = load_pdb(cfg.maps)

    else:
        df_pdb = load_seq_pdb(cfg.pep_set, cfg.hla_set, cfg.maps)
    
    # Calculate peptide length
    # df_pdb['peptide_len'] = df_pdb['peptide'].apply(lambda x: len(x))
    df_pdb['peptide_len'] = df_pdb['distance_matrix'].apply(lambda x: len(x))
    df_pdb = df_pdb[df_pdb['peptide_len'] < 16].reset_index(drop=True)  # remove 16-mer


    # Format labels
    df_pdb['distances'] = df_pdb['distance_matrix'].apply(format_dist)
    df_pdb['dihedrals'] = df_pdb['dihedral_matrix'].apply(format_angles)

    # Rescale labels
    df_pdb['scaled_distances'] = df_pdb['distances'].apply(scale_dist_log) #scale_dist_log
    # Each dihedrals cell ~ [9, 3] for phi, psi, omega
    df_pdb['scaled_dihedrals'] = df_pdb['dihedrals'].apply(scale_angles_separately)
    # df_pdb['phi'] = df_pdb['dihedrals'].apply(lambda x: x[:,0])
    # df_pdb['psi'] = df_pdb['dihedrals'].apply(lambda x: x[:,1])
    # phi = np.concatenate(df_pdb['phi'].apply(scale_angles_flip)) 
    # psi = np.concatenate(df_pdb['psi'].apply(scale_angles_flip)) 
    # df_pdb['scaled_dihedrals'] = 

    # Pad labels to max peptide length
    df_pdb['scaled_distances'] = df_pdb['scaled_distances'].apply(pad)
    df_pdb['scaled_dihedrals'] = df_pdb['scaled_dihedrals'].apply(pad)

    # Separate phi, psi
    df_pdb['scaled_phi'] = df_pdb['scaled_dihedrals'].apply(lambda x: x[:, 0])
    df_pdb['scaled_psi'] = df_pdb['scaled_dihedrals'].apply(lambda x: x[:, 1])

    # Flatten distances labels
    df_pdb['scaled_distances'] = df_pdb['scaled_distances'].apply(np.ravel)  # flatten [L_pep, 34] --> [L_pep*34]

    # Flag labels that have nans
    # Includes padded elements, elements with N/A phi/psi (at the edges),
    # and missing data
    df_pdb['flag_distances'] = df_pdb['scaled_distances'].apply(lambda x: np.isnan(x))
    df_pdb['flag_phi'] = df_pdb['scaled_phi'].apply(lambda x: np.isnan(x))
    df_pdb['flag_psi'] = df_pdb['scaled_psi'].apply(lambda x: np.isnan(x))

    # For weighting augmented vs. xtal structures
    # Save to s3

    df_pdb['label'] = 1  # they are all binders
    
    df_pdb['is_xtal_distances'] = df_pdb.apply(
        lambda row: np.ones(15*34) if len(row['pdbid']) < 5 else np.zeros(15*34),
        axis=1)
    df_pdb['is_xtal_phi'] = df_pdb.apply(
        lambda row: np.ones(15) if len(row['pdbid']) < 5 else np.zeros(15),
        axis=1)
    df_pdb['is_xtal_psi'] = df_pdb['is_xtal_phi']

    df_pdb['is_9mer_distances'] = df_pdb.apply(
        lambda row: np.ones(15*34) if row['peptide_len'] == 9 else np.zeros(15*34),
        axis=1
    )
    df_pdb['is_9mer_phi'] = df_pdb.apply(
        lambda row: np.ones(15) if row['peptide_len'] == 9 else np.zeros(15),
        axis=1
    )
    df_pdb['is_9mer_psi'] = df_pdb['is_9mer_phi']

    df_pdb = df_pdb[['peptide', 'hla_sequence', 'label', 'peptide_len',
                     'scaled_distances', 'scaled_phi', 'scaled_psi',
                     'flag_distances', 'flag_phi', 'flag_psi', 'pdbid', 'hla',
                     'is_xtal_distances', 'is_xtal_phi', 'is_xtal_psi',
                     'is_9mer_distances', 'is_9mer_phi', 'is_9mer_psi']]
    logger.info(df_pdb.dtypes)
    logger.info(f'Exporting shape: {df_pdb.shape}')
    df_pdb.to_parquet(cfg.dump_path, index=None)

if __name__ == '__main__':
    main()
    