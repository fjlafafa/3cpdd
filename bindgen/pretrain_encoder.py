import torch
import torch.nn as nn
import torch.nn.functional as F
from bindgen.sparse_encoder import EGNNEncoder, HierEGNNEncoder
from bindgen.utils import *
from bindgen.nnutils import *
from bindgen.data import make_batch
import os

# args should contain: input_dim, embed_dim, supervise_signal = ['ECnumber'])
# 


class PretrainEncoder(ABModel):
    """This aims to use a contrastive learning framework to embed the features into a latent space, the high level structural information should contain the relavent chemical property information"""
    def __init__(self, args):
        super(PretrainEncoder, self).__init__(args)
        assert isinstance(args.supervise_signal, list)
        self.signal = args.supervise_signal
        self.signal_num = len(self.signal)
        self.embed_dim = args.embed_dim
        self.input_dim = args.hidden_size
        
        # we will use three fully connected layers here because we anticipate the dataset size to be quite large.
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.ReLU(True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        
        self.hierarchical = args.hierarchical
        if args.hierarchical:
            self.struct_mpn = HierEGNNEncoder(args)
        else:
            self.struct_mpn = EGNNEncoder(args)

    def forward(self, X, S, A):
        S = self.embedding(S)
        V = self.features._dihedrals(X)
        h, _ = self.struct_mpn(X, V, S, A)
        # X: B, N, H
        global_features = h.mean(axis=1)
        return self.model(global_features)
    
    def save_model(self, e, args):
        ckpt = (self.model.state_dict(), self.struct_mpn.state_dict(), args)
        torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e}"))
            
class PretrainEncoderWrapper(nn.Module):
    def __init__(self, args, path='./ckpts/pretrain/model.ckpt.19', **kwarg):
        super().__init__()
        ckpt = torch.load(path)
        _, state_dict, args_pretrain = ckpt
        self.struct_mpn = EGNNEncoder(args_pretrain, **kwarg)
        self.struct_mpn.load_state_dict(state_dict, strict=False)
    
    def forward(self, X, V, S, A):
        return self.struct_mpn(X, V, S, A)