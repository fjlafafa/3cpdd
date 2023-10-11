import torch
import numpy as np
from collections import deque

class MocoContrastiveBuffer:
    def __init__(self, capacity, embed_dim, device, seed):
        self.device = device
        self.embedding = torch.zeros(capacity, embed_dim, dtype=torch.float).contiguous().pin_memory()
        self.EC = torch.zeros(capacity, dtype=torch.int).contiguous().pin_memory()
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.size = 0
        self.capacity = capacity

    def __repr__(self) -> str:
        return 'MocoBuffer'

    def add(self, embedding, EC):
        self.embedding[self.idx] = embedding.detach()
        self.EC[self.idx] = EC
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size, ec, embedding=None, choice='distance'):
        if choice == 'rng':
            sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
            batch = ()
            sample_idxs = torch.tensor(sample_idxs)
            embeddings = self.embedding[sample_idxs].to(self.device)
            return embeddings
        else:
            distance = torch.norm((self.embedding[:self.size].to(self.device) - embedding[None,:]), p=2, dim=1)
            idx = torch.argsort(distance)
            sample_idx = []
            real = embedding
            for i in range(self.size):
                if len(sample_idx) < batch_size and self.EC[idx[i]] != ec:
                    sample_idx.append(idx[i])
                elif self.EC[idx[i]] == ec:
                    real = self.embedding[idx[i]]
            sample_idx = torch.tensor(sample_idx)
            embeddings = self.embedding[sample_idx].to(self.device)
            return embeddings, real


