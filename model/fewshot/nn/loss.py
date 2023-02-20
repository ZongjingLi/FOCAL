import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import EPS

class FewshotLoss(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.tau = config.fewshot_tau

    def forward(self,outputs,inputs):
        losses = []
        for i,category in enumerate(inputs["val_sample"]["category"]):
            ends = outputs["val_sample"]["end"][i]