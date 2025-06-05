import torch
import torch.nn as nn
from argparse import Namespace
from tokenizer.tokenizer import MolTranBertTokenizer
from train_pubchem_light import LightningModule
from fast_transformers.masking import LengthMask as LM
import yaml

class MolFormer():
    # adapted from: https://github.com/IBM/molformer/blob/main/notebooks/pretrained_molformer/frozen_embeddings_classification.ipynb

    def __init__(self):
        with open('/hpc/home/yc583/Tahoe100M_practice/pretrained_molformer/hparams.yaml', 'r') as f:
            config = Namespace(**yaml.safe_load(f))
        self.tokenizer = MolTranBertTokenizer('/hpc/home/yc583/Tahoe100M_practice/pretrained_molformer/bert_vocab.txt')
        ckpt = '/hpc/home/yc583/Tahoe100M_practice/pretrained_molformer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
        self.model = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)

    def embed(self, smiles):
        self.model.eval()
        embeddings = []
        batch_enc = self.tokenizer.batch_encode_plus(smiles, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
        with torch.no_grad():
            token_embeddings = self.model.blocks(self.model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
        # average pooling over tokens
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        embeddings.append(embedding.detach().cpu())
        return torch.cat(embeddings)



class Transformer(nn.Module):
    def __init__(self, gene_emb_dim, num_genes, emb_dim, max_len, num_heads, 
                num_layers, dim_feedforward, dropout, pretrained_smiles_mdl,
    ):
        super(Transformer, self).__init__()

        self.emb_gene = nn.Embedding(num_genes, gene_emb_dim) # [batch_size, emb_dim]
        self.proj_expression = nn.Linear(1, gene_emb_dim)
        self.proj_gene_input = nn.Linear(gene_emb_dim, emb_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu"
            ),
            num_layers=num_layers
        )

        # pretrained smiles transformer
        # adapted from: https://github.com/IBM/molformer/blob/main/notebooks/pretrained_molformer/frozen_embeddings_classification.ipynb
        with open(pretrained_smiles_mdl + '/hparams.yaml', 'r') as f:
            config = Namespace(**yaml.safe_load(f))
        self.smiles_tokenizer = MolTranBertTokenizer(pretrained_smiles_mdl + '/bert_vocab.txt')
        ckpt = pretrained_smiles_mdl + '/N-Step-Checkpoint_3_30000.ckpt'
        self.smiles_encoder = LightningModule(config, self.smiles_tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=self.smiles_tokenizer.vocab)
        smiles_emb_dim = config['n_embd']

        self.proj_smiles_input = nn.Linear(smiles_emb_dim, emb_dim)

        self.out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

    def embed(self, smiles):
        self.smiles_encoder.eval()
        embeddings = []
        batch_enc = self.smiles_tokenizer.batch_encode_plus(smiles, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
        with torch.no_grad():
            token_embeddings = self.smiles_encoder.blocks(self.smiles_encoder.tok_emb(idx), length_mask=LM(mask.sum(-1)))
        # average pooling over tokens
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        embeddings.append(embedding.detach().cpu())
        return torch.cat(embeddings)

    def forward(self, 
                genes, # [batch_size, max_len]
                expressions, # [batch_size, max_len]
                smiles, # single string
                masks, # [batch_size, max_len] 1: unpadded, 0: padded
                ):

        genes_emb = self.emb_gene(genes)
        expressions = expressions.unsqueeze(-1) # [batch_size, max_len] -> [batch_size, max_len, 1]
        expressions_emb = self.proj_expression(expressions) # [batch_size, max_len, 1] -> [batch_size, max_len, emb_dim]
        gene_expr_emb = genes_emb + expressions_emb # [batch_size, max_len, emb_dim]

        # filter padded positions out
        masks = masks.unsqueeze(-1) # [batch_size, max_len] -> [batch_size, max_len, 1]
        pooled_gene_expr_emb = ((gene_expr_emb * masks).sum(dim=1))/(masks.sum(dim=1, keepdim=True)) # mean pooling: [batch_size, max_len, emb_dim] -> [batch_size, emb_dim]
        gene_features = self.proj_input(pooled_gene_expr_emb) # [batch_size, emb_dim]


        smiles_emb = self.embed(self.smiles).numpy()
        smiles_features = self.proj_smiles_input(smiles_emb) # [batch_size, emb_dim]

        features = torch.stack([gene_features, smiles_features], dim=0) # [2, batch_size, emb_dim]
        out = self.encoder(features) # [2, batch_size, emb_dim]
        out_pooled = out.mean(dim=0) # mean pooling: [2, batch_size, emb_dim] -> [batch_size, emb_dim]
        target_pred = self.regressor(out) # [batch_size, 1]

        return target_pred
        


        