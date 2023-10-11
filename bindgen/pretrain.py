import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os

from bindgen import *
from tqdm import tqdm
from copy import deepcopy
torch.set_num_threads(8)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='ckpts/pretrain')
    parser.add_argument('--load_model', default=None)

    parser.add_argument('--sparse_encoder', action='store_true', default=False)
    parser.add_argument('--no_progress_bar', action='store_true', default=False)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--cdr', default='3')
    parser.add_argument('--hierarchical', action='store_true', default=False)

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_tokens', type=int, default=100)
    parser.add_argument('--k_neighbors', type=int, default=9)
    parser.add_argument('--L_target', type=int, default=20)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--rstep', type=int, default=8)
    parser.add_argument('--clash_step', type=int, default=10)
    parser.add_argument('--vocab_size', type=int, default=21)
    parser.add_argument('--num_rbf', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay_step', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--clip_norm', type=float, default=1.0)

    parser.add_argument('--pretrain_path', default='out.jsonl')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--supervise_signal', default=['InfoNCE'])
    parser.add_argument('--embed_dim', default=128)
    parser.add_argument('--tau', default=1e-2)
    args = parser.parse_args()
    return args

def soft_update(args, target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - args.tau) * target_param.data + args.tau * source_param.data)
        
def loss(paired_data, loss_type='Triplet', signals = ['ECnumber']):
    loss = 0
    for signal in signals:
        if signal == "ECnumber":
            
            if loss_type == 'Triplet':
                anchor_embedding, positive_embedding, negative_embedding = paired_data
                positive_dist = torch.norm(anchor_embedding - positive_embedding, 2, dim = 1)
                negative_dist = nn.functional.normalize(anchor_embedding - negative_embedding, 2, dim = 1)
                loss += F.relu(positve_dist - negative_dist + 10).mean()
                
            elif loss_type == 'InfoNCE':
                anchor_embedding, positive_embedding, negatives_embedding = paired_data
                anchor_embedding = nn.functional.normalize(anchor_embedding, p=2, dim=0)
                positive_embedding = nn.functional.normalize(positive_embedding, p=2, dim=0)
                negative_list = torch.zeros((negatives_embedding.shape[0]))
                for i, negative_embedding in enumerate(negatives_embedding):
                    negative_embedding = nn.functional.normalize(negative_embedding, p=2, dim=0)
                    negative_list[i] = torch.dot(anchor_embedding, negative_embedding - positive_embedding)
                return max(negative_list)+torch.log(torch.sum(torch.exp(negative_list-max(negative_list))))
                    
            else:
                raise NotImplementedError("loss_type should be Triplet or SuperConHard")
                
        else:
            raise NotImplementedError("Signal should be among: ECnumber")
    return loss

if __name__ == '__main__':
    
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)

    data = ECSupervisedDataset(
        args.pretrain_path,
        not args.no_progress_bar   
    )

    loader = DataLoader(data, args.batch_size, collate_fn=data.collate_fn, shuffle = True)
    
    encoder = PretrainEncoder(args).cuda()
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, args.decay_step, 0.1)
    
    contrast_encoder = deepcopy(encoder).cuda()
    buffer = MocoContrastiveBuffer(capacity=100, embed_dim=args.embed_dim, device='cuda', seed=args.seed)
    for epoch in range(args.epochs):
        for batch in tqdm(loader):
            X, S, A, V, ECs = batch
            embedding = encoder(X, S, A)
            contrast_embedding = contrast_encoder(X, S, A)
            soft_update(args, contrast_encoder, encoder)
            for item, EC in zip(contrast_embedding, ECs):
                buffer.add(item, EC)
            
            if buffer.size == buffer.capacity:
                train_loss = 0
                for item, EC in zip(embedding, ECs):
                    embeddings, real = buffer.sample(batch_size=10, ec=ECs, embedding=item)
                    paired_data = (item, real, embeddings)
                    train_loss += loss(paired_data, loss_type='InfoNCE')
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        encoder.save_model(epoch, args)
        lr_scheduler.step()
    