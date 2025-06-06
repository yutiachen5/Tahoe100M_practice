import numpy as np
import argparse
import random
import os
import torch
import torch.nn as nn
from dataset import TahoeDataset
from models import Transformer
from collate import collate
from torch.utils.data import DataLoader, random_split
from pretrained_ChemBERT.smiles_emb import SmilesEmb
from functools import partial

def seed_everything(seed):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, train_loader, val_loader, num_epoches, use_cuda, optimizer, smiles_encoder):
    if use_cuda:
        model.cuda()
    criterion = nn.MSELoss()

    for i in range(1, num_epoches+1):
        model.train()
        train_loss_current = 0.
        for k, (genes, expressions, gene_masks, smiles, smiles_masks, targets) in enumerate(train_loader):
            if use_cuda:
                genes, expressions, gene_masks, smiles, smiles_masks, targets = \
                    genes.cuda(), expressions.cuda(), gene_masks.cuda(), smiles.cuda(), smiles_masks.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            preds = model(genes, expressions, smiles, gene_masks, smiles_masks, smiles_encoder)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss_current += loss.item()
        train_loss_final = train_loss_curent/len(train_loader)

        # validation
        model.eval()
        val_loss_current = 0.
        with torch.no_grad():
            for genes, expressions, smiles, targets, padding_masks in enumerate(val_loader):
                if use_cuda:
                    genes, expressions, smiles, targets, padding_masks = genes.cuda(), expressions.cuda(), smiles.cuda(), targets.cuda(), padding_masks.cuda()
                pred = model(genes, expressions, smiles, padding_masks)
                loss = criterion(preds, targets)
                val_loss_current += loss.item()
            val_loss_final = val_loss_current/len(val_loader)

        print(f"Epoch [{i}/{num_epochs}]  "f"Train Loss: {train_loss_final:.4f}  "f"Val Loss: {val_loss_final:.4f}")

    

def main():
    parser = argparse.ArgumentParser('Predicting IC50 based on gene, gene expression, and drug smiles.')

    parser.add_argument('--d-model', type=int, default=256, help='the number of expected features in the encoder/decoder inputs (default=256)')
    parser.add_argument('--dim-feedforward', type=int, default=512, help='the dimension of the feedforward network model (default=512)')
    parser.add_argument('--nhead', type=int, default=4, help='the number of heads in the multiheadattention models (default=4)')
    parser.add_argument('--nlayer', type=int, default=3, help='the number of sub-encoder-layers in the encoder (default=3)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability (default: 0.1)')
    parser.add_argument('--max-length', type=int, default=512, help='maximum length (default: 512)')
    parser.add_argument('--gene-emb-dim', type=int, default=128, help='embeding dimension of gene and gene expression (default:128)')

    parser.add_argument('--data', default='/hpc/home/yc583/Tahoe100M_practice/data/Tahoe_GDSC_processed.csv', help='path to training dataset')
    parser.add_argument('--pretrained-smiles-mdl', type=str, default='seyonec/ChemBERTa-zinc-base-v1', help='base path to pretrained smiles transformer')
    parser.add_argument('--val-size', type=float, default=0.2, help='size in percentage of validation (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=256, help='minibatch size (default: 256)')
    parser.add_argument('-n', type=int, default=100, help='number of training epoches (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('-o', help='output file path', default='/hpc/home/yc583/Tahoe100M_practice/saved_mdls/')
    parser.add_argument('--seed', help='random seed', type=int, default=1124)
    parser.add_argument('--name', type=str, default='test_run', help='name of the run for saving to wandb')
    parser.add_argument('-d', type=int, default=-2, help='compute device to use')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization (default: 1e-5)')

    args = parser.parse_args()

    seed_everything(args.seed)
    d = args.d
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    smiles_emb = SmilesEmb(pretrained_smiles_mdl = args.pretrained_smiles_mdl)
    smiles_tokenizer = smiles_emb.get_tokenizer()
    smiles_encoder, smiles_emb_dim = smiles_emb.get_encoder()

    TahoeData = TahoeDataset(args.data)
    print(f'loaded {len(TahoeData)} observations')
    num_genes = TahoeData.get_num_genes()
    print(f'number of unique genes: {num_genes}')

    train_size = int(len(TahoeData) * (1 - args.val_size))
    train_dataset, val_dataset = random_split(TahoeData, [train_size, len(TahoeData) - train_size])

    collate_fn = partial(collate, smiles_tokenizer=smiles_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        
    model = Transformer(gene_emb_dim=args.gene_emb_dim, num_genes=num_genes, emb_dim=args.d_model, num_heads=args.nhead, num_layers=args.nlayer, 
                        dim_feedforward=args.dim_feedforward, 
                        dropout=args.dropout, max_len=args.max_length, smiles_emb_dim=smiles_emb_dim,)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: ', total_params)
    print('number of trainable params: ', trainable_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train(model,  train_loader, val_loader, args.n, use_cuda, optimizer, smiles_encoder)

    save_path = args.output / 'ic50_predictor.pth'
    torch.save(model.state_dict(), save_path)
    print(f"model saved to {save_path}")

if __name__ == '__main__':
    main()

