import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
# from .data_utils import
from .loading_utils import load_transphla_data
from .vocab_dict import aa_dict

__all__ = ['HybridDataset']

class HybridDataset(ConcatDataset): # torch.utils.data.Dataset
    """Dataset of both sequences and structures (with distance, angle labels)
    """
    def __init__(self, is_train, Y_cols, ):
        """
        Parameters
        ----------
        TBD
        """
        super(HybridDataset).__init__()
        self.is_train = is_train
        self.Y_cols = Y_cols
        self.df = load_transphla_data().reset_index(drop=True)
        self.vocab = aa_dict

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = torch.tensor(row['label'])
        peptide_grid = torch.full(15, self.vocab['-']).long()
        peptide = torch.tensor([self.vocab[r] for r in row['peptide']]).long()
        peptide_grid[:len(peptide)] = peptide
        mhc_pseudo = torch.tensor([self.vocab[r] for r in row['HLA_sequence']])
        return dict(
            peptide=peptide_grid,
            HLA_sequence=mhc_pseudo,
            label=label,
        )

    def __len__(self):
        return len(self.df)

