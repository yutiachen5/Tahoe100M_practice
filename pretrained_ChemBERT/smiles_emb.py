import torch
from transformers import AutoTokenizer, AutoModel

# pretrained smiles transformer, retrieved from: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
class SmilesEmb:
    def __init__(self, pretrained_smiles_mdl):
        self.pretrained_smiles_mdl = pretrained_smiles_mdl

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.pretrained_smiles_mdl)

    def get_encoder(self):
        smiles_encoder = AutoModel.from_pretrained(self.pretrained_smiles_mdl)
        smiles_emb_dim = smiles_encoder.config.hidden_size
        return smiles_encoder, smiles_emb_dim

        