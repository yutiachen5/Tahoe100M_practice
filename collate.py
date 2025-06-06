import torch
from torch.nn.utils.rnn import pad_sequence

def collate(batch, smiles_tokenizer):
    genes, expressions, smiles, targets = zip(*batch)
    lengths = [len(g) for g in genes]
    padded_genes = pad_sequence(genes, batch_first=True, padding_value=0)
    padded_expressions = pad_sequence(expressions, batch_first=True, padding_value=0)
    gene_padding_masks = torch.tensor([[1]*l + [0]*(padded_genes.shape[1] - l) for l in lengths], dtype=torch.bool) # 1: unpadded, 0: padded

    smiles_tokens = smiles_tokenizer(smiles,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt',)
    smiles_input_ids = smiles_tokens['input_ids']
    smiles_attention_masks = smiles_tokens['attention_mask']

    return padded_genes, padded_expressions, gene_padding_masks, smiles_input_ids, smiles_attention_masks, list(targets)
