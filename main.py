import numpy as np
import time
import argparse
import random
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import TahoeDataset
from models import Transformer
from collate import collate
from torch.utils.data import DataLoader, random_split
from pretrained_ChemBERT.smiles_emb import SmilesEmb
from functools import partial
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])

def seed_everything(seed):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict(model, test_loader, use_cuda, smiles_encoder):
    all_preds = []; all_targets = []
    model.eval()
    with torch.no_grad():
        for k, (genes, expressions, gene_masks, smiles, smiles_masks, targets) in enumerate(test_loader):
            if use_cuda:
                genes, expressions, gene_masks, smiles, smiles_masks, targets = \
                    genes.cuda(), expressions.cuda(), gene_masks.cuda(), smiles.cuda(), smiles_masks.cuda(), targets.cuda()
            else: 
                model.cpu(); smiles_encoder.cpu()
            preds = model(genes, expressions, smiles, gene_masks, smiles_masks, smiles_encoder)
            all_preds.extend(preds.squeeze(-1).detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        pcc, _ = pearsonr(all_targets, all_preds)
        scc, _ = spearmanr(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        print(f"MSE: {mse:.4f}, PCC: {pcc:.4f}, SCC: {scc:.4f}")

def validate(model, val_loader, use_cuda, smiles_encoder):
    model.eval()
    val_loss_sum = 0.
    criterion = nn.MSELoss()
    with torch.no_grad():
        for k, (genes, expressions, gene_masks, smiles, smiles_masks, targets) in enumerate(val_loader):
            if use_cuda:
                genes, expressions, gene_masks, smiles, smiles_masks, targets = \
                    genes.cuda(), expressions.cuda(), gene_masks.cuda(), smiles.cuda(), smiles_masks.cuda(), targets.cuda()
            preds = model(genes, expressions, smiles, gene_masks, smiles_masks, smiles_encoder)
            loss = criterion(preds.squeeze(-1), targets)
            val_loss_sum += loss.item()
        val_loss = val_loss_sum/len(val_loader)
    return val_loss


def train(model, train_loader, val_loader, num_epochs, use_cuda, optimizer, smiles_encoder, args):
    if use_cuda:
        model.cuda()
        smiles_encoder.cuda()

    criterion = nn.MSELoss()
    min_val_loss = 1000
    epoch_no_improve = 0
    best_mdl = None

    for i in range(1, num_epochs+1):
        model.train()
        train_loss_sum = 0.
        for k, (genes, expressions, gene_masks, smiles, smiles_masks, targets) in enumerate(train_loader):
            if use_cuda:
                genes, expressions, gene_masks, smiles, smiles_masks, targets = \
                    genes.cuda(), expressions.cuda(), gene_masks.cuda(), smiles.cuda(), smiles_masks.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            preds = model(genes, expressions, smiles, gene_masks, smiles_masks, smiles_encoder)
            loss = criterion(preds.squeeze(-1), targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() 
        train_loss = train_loss_sum/len(train_loader)
        val_loss = validate(model, val_loader, use_cuda, smiles_encoder)

        print(f"Epoch [{i}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        wandb.log({'train loss':train_loss, 'validation loss':val_loss})

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_mdl = deepcopy(model)
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
        if epoch_no_improve >= args.patience:
            print(f"Training complete. Best validation loss: {min_val_loss:.4f}")
            break
    return best_mdl

    

def main():
    startTime = time.time()
    parser = argparse.ArgumentParser('Predicting IC50 based on gene, gene expression, and drug smiles.')

    parser.add_argument('--d-model', type=int, default=512, help='the number of expected features in the encoder/decoder inputs (default=256)')
    parser.add_argument('--dim-feedforward', type=int, default=512, help='the dimension of the feedforward network model (default=512)')
    parser.add_argument('--nhead', type=int, default=4, help='the number of heads in the multiheadattention models (default=4)')
    parser.add_argument('--nlayer', type=int, default=3, help='the number of sub-encoder-layers in the encoder (default=3)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability (default: 0.1)')
    parser.add_argument('--max-length', type=int, default=6000, help='maximum length (default: 512)')
    parser.add_argument('--gene-emb-dim', type=int, default=128, help='embeding dimension of gene and gene expression (default:128)')

    parser.add_argument('--data', default='/hpc/group/biostat/yc583/Tahoe_GDSC_0.2.csv.gz', help='path to training dataset')
    parser.add_argument('--patience', type=int, default=3, help='patience for early stopping (default: 3)')
    parser.add_argument('--pretrained-smiles-mdl', type=str, default='seyonec/ChemBERTa-zinc-base-v1', help='base path to pretrained smiles transformer')
    parser.add_argument('--val-size', type=float, default=0.15, help='size in percentage of validation and testing (default: 0.15)')
    parser.add_argument('--batch-size', type=int, default=256, help='minibatch size (default: 256)')
    parser.add_argument('-n', type=int, default=100, help='number of training epoches (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('-o', help='output file path', default='/hpc/home/yc583/Tahoe100M_practice/saved_mdls/')
    parser.add_argument('--seed', help='random seed', type=int, default=1124)
    parser.add_argument('--name', type=str, default='test_run', help='name of the run for saving to wandb')
    parser.add_argument('-d', type=int, default=-2, help='compute device to use')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization (default: 1e-5)')

    args = parser.parse_args()

    all_configs = [s for s in [100, 200, 300]] # 3 replicated runs with different random seeds
    args.seed = all_configs[taskID-3]
    args.name = 'run_s_'+str(args.seed)

    wandb.init(
        project="Tahoe",
        name=args.name, 
        config=vars(args) 
    )

    seed_everything(args.seed)

    d = args.d
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    smiles_emb = SmilesEmb(pretrained_smiles_mdl = args.pretrained_smiles_mdl)
    smiles_tokenizer = smiles_emb.get_tokenizer()
    smiles_encoder, smiles_emb_dim = smiles_emb.get_encoder()
    for param in smiles_encoder.parameters():
        param.requires_grad = False

    TahoeData = TahoeDataset(args.data)
    print(f'loaded {len(TahoeData)} observations', flush=True)
    num_genes = TahoeData.get_num_genes()
    print(f'number of unique genes: {num_genes}', flush=True)

    n_val = int(len(TahoeData)*args.val_size)
    train_dataset, val_test_dataset = random_split(TahoeData, [len(TahoeData) - 2*n_val, 2*n_val])
    val_dataset, test_dataset = random_split(val_test_dataset, [n_val, n_val])

    collate_fn = partial(collate, smiles_tokenizer=smiles_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        
    model = Transformer(gene_emb_dim=args.gene_emb_dim, num_genes=num_genes, emb_dim=args.d_model, num_heads=args.nhead, num_layers=args.nlayer, 
                        dim_feedforward=args.dim_feedforward, 
                        dropout=args.dropout, max_len=args.max_length, smiles_emb_dim=smiles_emb_dim,)
    print(model, flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: ', total_params)
    print('number of trainable params: ', trainable_params)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.95)

    print('training...')
    best_mdl = train(model,  train_loader, val_loader, args.n, use_cuda, optimizer, smiles_encoder, args)
    print('predicting...')
    predict(best_mdl, test_loader, use_cuda, smiles_encoder)

    save_path = args.o + args.name+'_ic50_predictor.pth'
    torch.save(model.state_dict(), save_path)
    print(f"model saved to {save_path}")

    endTime = time.time()
    print("Elapsed time:", round(endTime-startTime/3600,1), "hours")

if __name__ == '__main__':
    main()

