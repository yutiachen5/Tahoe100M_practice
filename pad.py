import torch
from torch.nn.utils.rnn import pad_sequence

def pad_genes(batch):
    genes, expressions, smiles, targets = zip(*batch)
    lengths = [len(g) for g in genes]
    padded_genes = pad_sequence(genes, batch_first=True, padding_value=0)
    padded_expressions = pad_sequence(expressions, batch_first=True, padding_value=0)
    padding_masks = torch.tensor([[1]*l + [0]*(padded_genes.shape[1] - l) for l in lengths], dtype=torch.bool) # 1: unpadded, 0: padded

    return padded_genes, padded_expressions, smiles, targets, padding_masks
