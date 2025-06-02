import torch
from torch.utils.data import Dataset
# from .data_utils import
from philia.trainval_data.loading_utils import load_transphla_data
from philia.trainval_data import tokenize_single_seq


class SequenceDataset(Dataset):  # torch.utils.data.Dataset
    """Dataset of sequences
    """
    def __init__(self):
        """
        Parameters
        ----------
        TBD
        """
        # super(SequenceDataset).__init__()
        self.df = load_transphla_data().reset_index(drop=True)
        self.df = self.df.rename(columns={'HLA_sequence': 'hla_sequence'})

    def configure_alphabet(self, alphabet_hla, alphabet_peptide):
        self.alphabet_hla = alphabet_hla
        self.alphabet_peptide = alphabet_peptide

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # import pdb; pdb.set_trace()
        label = torch.tensor(row['label']).float()
        # peptide = torch.full([15], self.vocab['-']).long()
        # peptide_true_length = torch.tensor([self.vocab[r] for r in row['peptide']]).long()
        # peptide_len = len(peptide_true_length)
        # peptide[:peptide_len] = peptide_true_length
        peptide = tokenize_single_seq(self.alphabet_peptide, row['peptide'], 15)
        mhc_pseudo = tokenize_single_seq(self.alphabet_hla, row['hla_sequence'])
        peptide_len = len(row['peptide'])
        return dict(
            peptide=peptide,
            peptide_len=torch.tensor(peptide_len).long(),
            hla_sequence=mhc_pseudo,
            label=label,
            loss_weight=1.0/(label*4 + 1),
        )

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    import esm
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    dataset = SequenceDataset()
    dataset.configure_alphabet(
        alphabet_hla=alphabet, alphabet_peptide=alphabet)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=7)
    for b in loader:
        print(b)
        print(b['peptide'].shape, b['hla_sequence'].shape, b['label'].shape)
        break
