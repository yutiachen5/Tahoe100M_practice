import pandas as pd
from torch.utils.data import Dataset


class TahoeDataset(dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        df['genes'] = df['genes'].apply(lambda x: x[1:]) # exclude marker token
        df['expression'] = df['expressions'].apply(lambda x: x[1:])
        self.genes = df['genes']
        self.expressions = df['expressions']
        self.smils = df['canonical_smiles']
        self.targets = df['LN_IC50']

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, i):
        gene = self.genes[i]
        expression = self.expressions[i]
        smiles = self.smiles[i]
        target = self.targets[i]

        return gene, expression, smiles, target

    def get_num_genes(self):
        unique_genes = []
        for i in range(len(self.genes)):
            unique_genes.extend(self.genes[i])
        
        return len(set(unique_gene)) 