import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module): # adpated from: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1) # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, gene_emb_dim, num_genes, emb_dim, max_len, num_heads, 
                num_layers, dim_feedforward, dropout, smiles_emb_dim,
    ):
        super(Transformer, self).__init__()

        self.emb_gene = nn.Embedding(num_genes, gene_emb_dim) # [batch_size, emb_dim]
        self.proj_expression = nn.Linear(1, gene_emb_dim)
        self.proj_gene_input = nn.Linear(gene_emb_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len)
        

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers
        )

        self.proj_smiles_input = nn.Linear(smiles_emb_dim, emb_dim)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim//2, 1)
        )

    def forward(self, 
                genes, # [batch_size, max_len]
                expressions, # [batch_size, max_len]
                smiles, # [batch_size, max_len]
                gene_masks, # [batch_size, max_len] 1: unpadded, 0: padded
                smiles_masks, # [batch_size, max_len] 1: unpadded, 0: padded
                smiles_encoder,
                ):

        # gene features
        genes_emb = self.emb_gene(genes)
        expressions = expressions.unsqueeze(-1) # [batch_size, max_len] -> [batch_size, max_len, 1]
        expressions_emb = self.proj_expression(expressions) # [batch_size, max_len, 1] -> [batch_size, max_len, gene_emb_dim]
        gene_expr_emb = genes_emb + expressions_emb # [batch_size, max_len, gene_emb_dim]
        # filter padded positions out
        pooled_gene_expr_emb = ((gene_expr_emb * gene_masks.unsqueeze(-1)).sum(dim=1))\
                                /(gene_masks.sum(dim=1, keepdim=True)) # mean pooling: [batch_size, max_len, gene_emb_dim] -> [batch_size, gene_emb_dim]
        gene_features = self.proj_gene_input(pooled_gene_expr_emb) # [batch_size, emb_dim]


        # smiles features
        smiles_emb = smiles_encoder(input_ids=smiles, 
                                    attention_mask=smiles_masks).last_hidden_state # smile_emb_dim=768
        smiles_emb = smiles_emb[:, 0, :] # CLS token to summarize the emb, [batch_size, smiles_emb_dim]
        smiles_features = self.proj_smiles_input(smiles_emb) # [batch_size, emb_dim]

        features = torch.stack([gene_features, smiles_features], dim=1) # [batch_size, 2, emb_dim]
        # positional encoding
        features = self.pos_encoder(features)
        out = self.encoder(features) # [batch_size, 2, emb_dim]
        out_pooled = out.mean(dim=1) # mean pooling: [batch_size, 2, emb_dim] -> [batch_size, emb_dim]
        pred = self.regression_head(out_pooled) # [batch_size, 1]

        return pred
        


        