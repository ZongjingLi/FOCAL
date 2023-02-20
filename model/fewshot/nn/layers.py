import torch
import torch.nn as nn

from model.nn.mlp import MLP
from utils.tensor  import unbind, bind

class MessagePassing(nn.Module):
    def __init__(self,dim,in_channels,mid_channels,out_channels,n_edge_types):
        super().__init__()
        self.dim = dim
        self.width = in_channels // 2
        self.out_channels = out_channels
        self.mlp = MLP(in_channels,mid_channels,out_channels * n_edge_types, bias = False)

    