from typing import Union, Tuple, List
from itertools import repeat
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math

from torchsparse import SparseTensor
from torchsparse.backbones.unet import SparseResUNet
from torchsparse.utils.collate import sparse_collate


@torch.no_grad()
def ravel_hash(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, x.size()

    x = (x - torch.min(x, dim=0)[0]).long()
    xmax = torch.max(x, dim=0)[0] + 1

    h = x.new_zeros(x.size(0), dtype=torch.long)
    for k in range(x.size(1) - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    assert h.isfinite().all(), "hash is exploding!"
    return h

@torch.no_grad()
def unique(x: torch.Tensor, dim=0):
    """Reference: https://github.com/pytorch/pytorch/issues/36748"""
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    return unique, index, inverse

def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_

def process(x: torch.Tensor):
    torch.save(x, "./debug.pth")
    return "tensors saved to ./debug.pth"

@torch.no_grad()
def sparse_quantize(coords_: torch.Tensor,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False) -> List[torch.Tensor]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = coords_.new_tensor(voxel_size)
    coords = torch.floor(coords_ / voxel_size).long() # .int()

    _, indices, inverse_indices = unique(ravel_hash(coords))
    assert coords.size(0) == len(indices), f"Coordinate collision. {process([coords, coords_])}"
    assert (coords < 2147483648).all(), f"{process([coords, coords_])}"
    coords = coords.int()
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

def sparse2dense(x: torch.Tensor, indices: torch.Tensor, masks: torch.Tensor, recover_check=False):
    """
    convert sparse features to dense ones. 
    x: L, C
    indices: L
    mask: B, N
    """
    assert x.ndim == 2, x.size()
    assert indices.ndim == 1, indices.size()
    assert masks.ndim == 2, masks.size()
    assert x.size(0) == indices.size(0)
    
    B, N = masks.size()
    max_len = max([(indices == k).sum() for k in range(B)])
    assert 0 < max_len <= N
    dense_x = []
    for k in range(B):
        mask_x = indices == k
        length = mask_x.sum()
        if recover_check:
            assert length == masks[k].sum(), (length, masks[k].sum())
        assert length >= 0 and length <= max_len, f"{length} {max_len}"
        dense_row = F.pad(x[mask_x], (0, 0, 0, max_len-length))
        dense_x.append(dense_row)
    dense_x = torch.stack(dense_x, 0)
    return dense_x

class PointCloudUNet(SparseResUNet):
    def __init__(self, voxel_size, **kwargs) -> None:
        super().__init__(
            stem_channels=32,
            encoder_channels=[32, 64, 128, 256],
            decoder_channels=[256, 128, 96, 96], # Linear 96 -> 256
            **kwargs)
        self.linear = nn.Linear(96, 256)
        self.voxel_size = voxel_size
    
    def forward(self, X: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor) -> List[SparseTensor]:
        """
        Parameters
        ------------
        X: point cloud (B, N, 3)
        feats: (B, N, C)
        mask: padding mask for X, (B, N)
        
        Return
        ---------------
        List of tensors, output.F of every layer. The last layer is (B, N, C)
        """
        mask = mask.bool()
        
        inputs = []
        for pc, feat, m in zip(X, feats, mask):
            pc = pc[m]
            if pc.size(0) == 0:
                inputs.append(SparseTensor(coords=pc.int(), feats=feat[m].float()).to(pc.device))
                print("Empty pc encountered. ")
            else:
                # pca
                pc = PCA(n_components=3).to(pc.device).fit_transform(pc)
                pc = pc - pc.min(dim=0, keepdim=True)[0].detach()
                
                D = torch.sum((pc[:, None] - pc[None, :]) ** 2, dim=-1).sqrt().detach()
                index = torch.arange(pc.size(0), device=pc.device)
                D[index, index] = 100
                assert D.min() > 0
                if D.min() < 2 * math.sqrt(3) * self.voxel_size:
                    pc = pc * (2 * math.sqrt(3) * self.voxel_size / D.min())
                # print(pc.max(dim=1, keepdim=True)[0].detach())
                # print(pc.min(dim=1, keepdim=True)[0].detach())
                # print("coords", D.min())
                # if D.min() < math.sqrt(3) * self.voxel_size:
                #     # print((math.sqrt(3) * self.voxel_size / D.min()))
                #     pc = pc * (math.sqrt(3) * self.voxel_size / D.min())
            
                coords, indices = sparse_quantize(pc, self.voxel_size, return_index=True)
                feat = feat[indices].float()
                input = SparseTensor(coords=coords, feats=feat).to(coords.device)
                inputs.append(input)
        inputs = sparse_collate(inputs)

        # forward
        outputs = super().forward(inputs)
        
        out_feats = [sparse2dense(o.F, o.C[:, -1], mask, recover_check=(i==8)) for i, o in enumerate(outputs)] # B, N, C
        last_feats = self.linear(out_feats[-1])
        assert last_feats.size(1) <= mask.size(1)
        if last_feats.size(1) < mask.size(1):
            last_feats = F.pad(last_feats, (0, 0, 0, mask.size(1) - last_feats.size(1)))
        out_feats[-1] = last_feats
        
        return out_feats

def main() -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model: nn.Module = PointCloudUNet(in_channels=4, width_multiplier=1.0)
    model = model.to(device).train()

    # generate data
    input_size = 10000
    inputs = nn.Parameter((torch.rand(2, input_size, 4, device=device) - 0.5) * 200)
    pcs, feats = inputs[..., :3], inputs
    masks = torch.ones(2, input_size, dtype=torch.bool, device=device)
    masks[1, 7000:] = 0

    # forward
    outputs = model(pcs, feats, masks)

    # print feature shapes
    for k, output in enumerate(outputs):
        print(f'output[{k}].F.shape = {output.size()}')
    
    loss = sum([o.mean() for o in outputs])
    loss.backward()
        
    print(f'inputs.grad = {inputs.grad.abs().max()}')


if __name__ == '__main__':
    main()
