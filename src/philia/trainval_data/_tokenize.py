import torch

# aa_dict = {'C': 1, 'W': 2, 'V': 3, 'A': 4, 'H': 5, 'T': 6, 'E': 7, 'K': 8,
#               'N': 9, 'P': 10, 'I': 11, 'L': 12, 'S': 13, 'D': 14, 'G': 15,
#               'Q': 16, 'R': 17, 'Y': 18, 'F': 19, 'M': 20, '-': 0}

def tokenize_single_seq(alphabet, sequence: str, max_seq_len: int = 0):
    """
    if prepend_bos, append_eos:
    <cls> <encoded_seq> <eos> <pad>... until reaches max_seq_len+2

    """
    seq_len = len(sequence)
    seq_grid = torch.empty(
                (max(max_seq_len, seq_len)
                 + int(alphabet.prepend_bos)
                 + int(alphabet.append_eos),
                ),
                dtype=torch.int64,
            )
    seq_grid.fill_(alphabet.padding_idx)
    if alphabet.prepend_bos:
        seq_grid[0] = alphabet.cls_idx
    seq_encoded = alphabet.encode(sequence)
    seq = torch.tensor(seq_encoded, dtype=torch.int64)
    seq_grid[
        int(alphabet.prepend_bos) : len(seq_encoded) + int(alphabet.prepend_bos)
        ] = seq
    if alphabet.append_eos:
        seq_grid[len(seq_encoded) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return seq_grid
