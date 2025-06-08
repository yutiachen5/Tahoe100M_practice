import pandas as pd
import torch
from torch.utils.data import Dataset

class TahoeDataset(Dataset):
    def __init__(self, path):
        if path.endswith('.gz'):
            df = pd.read_csv(path, compression='gzip')
        else:
            df = pd.read_csv(path)
        
        df = df[['genes', 'expressions','canonical_smiles','LN_IC50']]
        df['genes'] = df['genes'] 
        df['expression'] = df['expressions']
        self.genes = df['genes']
        self.expressions = df['expressions']
        self.smiles = df['canonical_smiles']
        self.targets = df['LN_IC50']
    
        unique_genes = []
        for i in range(len(self.genes)):
            unique_genes.extend(eval(self.genes[i]))
        self.ids = sorted(set(unique_genes))
        self.gene_idx_map = {gene : i for i, gene in enumerate(self.ids)}

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, i):
        gene = torch.tensor([self.gene_idx_map[idx] for idx in eval(self.genes[i])], dtype=torch.long)
        expression = torch.tensor([float(x) for x in eval(self.expressions[i])], dtype=torch.float32)
        smiles = self.smiles[i]
        target = self.targets[i]

        return gene, expression, smiles, target

    def get_num_genes(self):
        return len(self.ids)