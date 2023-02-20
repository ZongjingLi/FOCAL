import torch
import torch.nn as nn

from utils import EPS
from .mlp import MLP
from .resnet import make_resnet_layers

class FeatureExtractor(nn.Module):
    def __init__(self,config):
        super().__init__()