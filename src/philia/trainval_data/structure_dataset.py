import torch
from torch.utils.data import Dataset
import esm
from philia.trainval_data.loading_utils import load_pdb_data, load_pdb_data_log #, load_pdb_data_log_flip
from philia.trainval_data import tokenize_single_seq


class StructureDataset(Dataset):  # torch.utils.data.Dataset
    """Dataset of structures (with distance, angle labels)
    """
    def __init__(self, version, scale_option):
        """
        Parameters
        ----------
        TBD
        """
        super(StructureDataset).__init__()
        self.version = version
        self.scale_option = scale_option
        if self.scale_option == 'scalar':
            self.df = load_pdb_data(self.version).reset_index(drop=True)
        elif self.scale_option == 'log':
            self.df = load_pdb_data_log(self.version).reset_index(drop=True)
        # elif self.scale_option == 'log_flip':
        #     self.df = load_pdb_data_log_flip(self.version).reset_index(drop=True)
        # self.valset = load_val_data(self.version)

    def configure_alphabet(self, alphabet_hla, alphabet_peptide):
        self.alphabet_hla = alphabet_hla
        self.alphabet_peptide = alphabet_peptide

    def __getitem__(self, index):
        row = self.df.iloc[index]
        peptide = tokenize_single_seq(self.alphabet_peptide, row['peptide'], 15)
        mhc_pseudo = tokenize_single_seq(self.alphabet_hla, row['hla_sequence'])
        peptide_len = len(row['peptide'])
        # peptide = row['peptide'] + '<mask>'*(15 - len(row['peptide']))
        # peptide = torch.tensor(self.alphabet.encode(peptide))
        # mhc_pseudo = torch.tensor(self.alphabet.encode(row['hla_sequence']))
        # peptide = torch.tensor([self.alphabet[r] for r in row['peptide']])
        # mhc_pseudo = torch.tensor([self.alphabet[r] for r in row['hla_sequence']])
        # Do not use from_numpy (non-writeable arrays)
        distances = torch.tensor(row['scaled_distances'].reshape(15, 34))  # [15, 34]
        phi = torch.tensor(row['scaled_phi'])  # [pep_L,]
        psi = torch.tensor(row['scaled_psi'])  # [pep_L,]
        flag_distances = torch.tensor(row['flag_distances'].reshape(15, 34)).bool()
        flag_phi = torch.tensor(row['flag_phi']).bool()
        flag_psi = torch.tensor(row['flag_psi']).bool()
        is_xtal_distances = torch.tensor(row['is_xtal_distances'].reshape(15, 34))
        is_xtal_phi = torch.tensor(row['is_xtal_phi'])
        is_xtal_psi = torch.tensor(row['is_xtal_psi'])
        is_9mer_distances = torch.tensor(row['is_9mer_distances'].reshape(15,34))
        is_9mer_phi = torch.tensor(row['is_9mer_phi'])
        is_9mer_psi = torch.tensor(row['is_xtal_psi'])
        peptide_len = len(row['peptide'])
        
        return dict(
            # pdbid=torch.tensor(index).long(),
            distances=distances,
            phi=phi,
            psi=psi,
            peptide=peptide,
            hla_sequence=mhc_pseudo,
            peptide_len=torch.tensor(peptide_len).long(),
            flag_distances=flag_distances,
            flag_phi=flag_phi,
            flag_psi=flag_psi,
            is_xtal_distances=is_xtal_distances,
            is_xtal_phi=is_xtal_phi,
            is_xtal_psi=is_xtal_psi,
            is_9mer_distances=is_9mer_distances,
            is_9mer_phi=is_9mer_phi,
            is_9mer_psi=is_9mer_psi,
            # label=torch.tensor([1]),  # all binders
        )

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    import esm
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    data = StructureDataset()
    data.configure_alphabet(
        alphabet_hla=alphabet, alphabet_peptide=alphabet)
    print(data.df.columns.values)
    example = data[0]
    print(example['distances'].shape)  # [15, 34]
    print(example['psi'].shape)  # [15,]
    print(example['peptide_len'].shape)  # []
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=7)
    for b in loader:
        # print(b)
        print(b['peptide'].shape, b['hla_sequence'].shape)  # [7, 15], [7, 34]
        print(b['flag_distances'].shape, b['flag_phi'].shape)  # [7, 15, 34], [7, 15]
        print(b['distances'].shape, b['phi'].shape, b['peptide_len'].shape)
        # ~ [7, 15, 34], [7, 15], [7]
        break
