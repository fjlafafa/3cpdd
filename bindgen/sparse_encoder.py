import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import copy

import os, sys
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from bindgen.utils import *
from bindgen.nnutils import *
from bindgen.data import ALPHABET, ATOM_TYPES
from bindgen.protein_features import ProteinFeatures
from bindgen.sparse_conv import PointCloudUNet
from bindgen.encoder import EGNNEncoder as EGNN_orig
import argparse

class EGNNEncoder(nn.Module):
    
    def __init__(self, args, node_hdim=0, features_type='backbone', update_X=True):
        super(EGNNEncoder, self).__init__()
        self.update_X = update_X
        self.features_type = features_type
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type=features_type,
                direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions[features_type]
        self.node_in += node_hdim
        
        if self.features_type == 'backbone':
            self.embedding = AAEmbedding()
            self.U_i = nn.Linear(self.embedding.dim(), args.hidden_size)
        
        if self.update_X:
            self.W_x = nn.Linear(args.hidden_size, args.hidden_size)
            self.U_x = nn.Linear(args.hidden_size, args.hidden_size)
            self.T_x = nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_size, 14))
        
        # self.atom_embedding = nn.Embedding(len(ATOM_TYPES), args.hidden_size)
        self.unet = PointCloudUNet(in_channels=self.node_in + args.hidden_size+3, width_multiplier=1.0, voxel_size=args.voxel_size)

        # for param in self.parameters():
        #     if param.dim() > 1:
        #         nn.init.xavier_uniform_(param)
    # N: number of aa, L: length of aa
    # [backbone] X: [B,N,L,3], V/S: [B,N,H], A: [B,N,L]
    # [atom] X: [B,N*L,3], V/S: [B,N*L,H], A: [B,N*L]
    def forward(self, X: Tensor, V: Tensor, S: Tensor, A: Tensor):
        """
        Similar to Graph CNNs.
        Infered from data.py. 
        X: coordinates of atoms [B, N * L, 3]
        V: node features [B, N*L, H]
        S: atom embeddings [B, N*L, H]
        A: atom types (one-hot encoding) [B, N * L]
        """
        # print(X.shape, V.shape, S.shape, A.shape)
        if self.features_type == 'backbone':
            S = self.U_i(S)
            
        mask = A.clamp(max=1).float() # Masks for all atoms (0 is for not an atom)
        vmask = mask[:,:,1] if self.features_type == 'backbone' else mask
        # feats = torch.cat([X, self.atom_embedding(A) * mask.unsqueeze(-1)], dim=-1)
        coords = X
        
        if self.features_type == 'backbone':
            # feats = feats.sum(dim=2) / mask.sum(dim=2).unsqueeze(-1)
            coords = coords[:, :, 1]
        
        feats = torch.cat([coords, V, S], dim=-1)
        # print(feats.shape)
        
        output_ = self.unet(coords, feats, vmask)
        # for index,i in enumerate(output_):
        #     print(index, i.shape)
        h = output_[-1]

        if self.update_X and self.features_type == 'backbone':
            ca_mask = mask[:,:,1]  # [B, N]
            mij = self.W_x(h).unsqueeze(2) + self.U_x(h).unsqueeze(1)  # [B,N,N,H], The force coefficient (Fij = Tx(mij) * (xi-xj))
            xij = X.unsqueeze(2) - X.unsqueeze(1)  # [B,N,N,L,3]
            xij = xij * self.T_x(mij).unsqueeze(-1)  # [B,N,N,L,3]
            f = torch.sum(xij * ca_mask[:,None,:,None,None], dim=2)  # [B,N,N,L,3] * [B,1,N,1,1]
            f = f / (1e-6 + ca_mask.sum(dim=1)[:,None,None,None])    # [B,N,L,3] / [B,1,1,1]
            X = X + f.clamp(min=-20.0, max=20.0)

        return h, X * mask[...,None]


class HierEGNNEncoder(nn.Module):

    def __init__(self, args, update_X=True, backbone_CA_only=True):
        super(HierEGNNEncoder, self).__init__()
        self.update_X = update_X
        self.backbone_CA_only = backbone_CA_only
        self.clash_step = args.clash_step
        self.residue_mpn = EGNNEncoder(
                args, features_type='backbone',
                node_hdim=args.hidden_size,
                update_X=False,
        )
        self.atom_mpn = EGNN_orig(
                args, features_type='atom',
                node_hdim=args.hidden_size,
                update_X=False,
        )
        if self.update_X:
            # backbone coord update
            self.W_x = nn.Linear(args.hidden_size, args.hidden_size)
            self.U_x = nn.Linear(args.hidden_size, args.hidden_size)
            self.T_x = nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_size, 4))
            # side chain coord update
            self.W_a = nn.Linear(args.hidden_size, args.hidden_size)
            self.U_a = nn.Linear(args.hidden_size, args.hidden_size)
            self.T_a = nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_size, 1))

        self.embedding = nn.Embedding(len(ATOM_TYPES), args.hidden_size) # Learnable embeddings (of length H) for different atom types
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # X: [B,N,L,3], V: [B,N,6], S: [B,N,H], A: [B,N,L]
    def forward(self, X:Tensor, V:Tensor, S:Tensor, A:Tensor):
        B, N, L = X.size()[:3]
        X_atom = X.view(B, N*L, 3)
        mask = A.clamp(max=1).float() # Masks for all atoms (0 is for not an atom)

        # atom message passing
        h_atom = self.embedding(A).view(B, N*L, -1) # [B, N*L, H]
        h_atom, _ = self.atom_mpn(X_atom, h_atom, h_atom, A.view(B,-1))
        h_atom = h_atom.view(B,N,L,-1)
        h_atom = h_atom * mask[...,None] # [B, N, L, H]
        h_A = h_atom.sum(dim=-2) / (1e-6 + mask.sum(dim=-1)[...,None]) # [B, N, H]

        # residue message passing
        # the atom features serves as additional residue features
        h_V = torch.cat([V, h_A], dim=-1) # [B, N, 6+H]
        h_res, _ = self.residue_mpn(X, h_V, S, A) # [B, N, H]

        if self.update_X:
            # backbone update (Eq. 10)
            bb_mask = mask[:,:,:4]  # [B, N, 4]
            X_bb = X[:,:,:4]  # backbone atoms (N, Cα, C, O)
            mij = self.W_x(h_res).unsqueeze(2) + self.U_x(h_res).unsqueeze(1)  # [B,N,N,H]
            xij = X_bb.unsqueeze(2) - X_bb.unsqueeze(1)  # [B,N,N,4,3]
            dij = xij.norm(dim=-1)  # [B,N,N,4]
            fij = torch.maximum(self.T_x(mij), 3.8 - dij)  # break term [B,N,N,4]
            xij = xij * fij.unsqueeze(-1)
            f_res = torch.sum(xij * bb_mask[:,None,:,:,None], dim=2)  # [B,N,N,4,3] * [B,1,N,4,1] -> [B,N,4,3]
            f_res = f_res / (1e-6 + bb_mask.sum(dim=1, keepdims=True)[...,None])  # [B,N,4,3]
            X_bb = X_bb + f_res.clamp(min=-20.0, max=20.0)

            # Clash correction
            # We assume if the pairwise distance dij is less than 3.8, there is a clash and some addtional force fij will push them apart. 
            for _ in range(self.clash_step):
                xij = X_bb.unsqueeze(2) - X_bb.unsqueeze(1)  # [B,N,N,4,3]
                dij = xij.norm(dim=-1)  # [B,N,N,4]
                fij = F.relu(3.8 - dij)  # repulsion term [B,N,N,4]
                xij = xij * fij.unsqueeze(-1)
                f_res = torch.sum(xij * bb_mask[:,None,:,:,None], dim=2)  # [B,N,N,4,3] * [B,1,N,4,1] -> [B,N,4,3]
                f_res = f_res / (1e-6 + bb_mask.sum(dim=1, keepdims=True)[...,None])  # [B,N,4,3]
                X_bb = X_bb + f_res.clamp(min=-20.0, max=20.0)

            # side chain update
            # Compute the force exerted on each atom within a residue. (Eq. 11)
            mij = self.W_a(h_atom).unsqueeze(3) + self.U_a(h_atom).unsqueeze(2)  # [B,N,L,1,H] + [B,N,1,L,H]
            xij = X.unsqueeze(3) - X.unsqueeze(2)  # [B,N,L,1,3] - [B,N,1,L,3] = [B,N,L,L,3]
            dij = xij.norm(dim=-1)  # [B,N,L,L]
            fij = torch.maximum(self.T_a(mij).squeeze(-1), 1.5 - dij)  # break term [B,N,L,L]
            xij = xij * fij.unsqueeze(-1)  # [B,N,L,L,3]
            f_atom = torch.sum(xij * mask[:,:,None,:,None], dim=3)  # [B,N,L,L,3] * [B,N,1,L,1] -> [B,N,L,3]
            X_sc = X + 0.1 * f_atom

            # Acquire the final coordinates
            # Note that Cα is the second atom in each residue
            if self.backbone_CA_only:
                X = torch.cat((X_sc[:,:,:1], X_bb[:,:,1:2], X_sc[:,:,2:]), dim=2)
                # Cα is fixed, while others moves with the side chain update
                # Using X_bb fixes that atom, and using X_sc takes the updated coordinates
            else:
                X = torch.cat((X_bb[:,:,:4], X_sc[:,:,4:]), dim=2)
                # All 4 backbone atoms are fixed, while others moves with the side chain update

        return h_res, X * mask[...,None]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/rabd/train_data.jsonl')
    parser.add_argument('--val_path', default='data/rabd/val_data.jsonl')
    parser.add_argument('--test_path', default='data/rabd/test_data.jsonl')
    parser.add_argument('--save_dir', default='ckpts/tmp')
    parser.add_argument('--load_model', default=None)

    parser.add_argument('--sparse_encoder', action='store_true', default=False)
    parser.add_argument('--cdr', default='3')
    parser.add_argument('--no_target', action='store_true', default=False)
    parser.add_argument('--att_refine', action='store_true', default=False)
    parser.add_argument('--hierarchical', action='store_true', default=False)
    parser.add_argument('--sequence', action='store_true', default=False)

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_tokens', type=int, default=100)
    parser.add_argument('--k_neighbors', type=int, default=9)
    parser.add_argument('--L_target', type=int, default=20)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--clash_step', type=int, default=10)
    parser.add_argument('--num_rbf', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--clip_norm', type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model = EGNNEncoder(args).cuda()
    
    X = torch.rand([2, 1164, 14, 3]).cuda() * 10
    V = torch.ones([2, 1164, 6]).cuda()
    S = torch.ones([2, 1164, 256]).cuda()
    A = torch.ones([2, 1164, 14]).cuda()
    
    model(X, V, S, A)