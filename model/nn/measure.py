import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import EPS

class Measure(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dim = config.concept_dim
        self.temperature = config.temperature

    def forward(self,x):
        return torch.sum(torch.log(self.softplus(x[..., self.dim:])), dim=-1)

    def measure_along_axis(self,x):
        return self.softplus(x[...,self.dim:])

    @classmethod
    def log2logit(cls,log):
        log = torch.clamp(log,max = EPS)
        logit = log - torch.log(1 - torch.exp(log))
        return logit

    def intersection(self,x,y):
        x_center, x_offset = x.chunk(2,-1)
        y_center, y_offset = y.chunk(2,-1)

        maxima = torch.min(x_center + x_offset,y_center + y_offset)
        minima = torch.max(x_center - x_offset,y_center - y_offset)

        intersection = torch.cat([maxima + minima, maxima - minima],-1)/2
        return intersection 