import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nn import build_entailment
from utils import underscores, freeze


class FewshotProgramExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self,config):
        super().__init__()
        network = self.NETWORK_REGISTRY(config.name)(config)
        entailment = build_entailment(config)
        self.learner = PipelineLearner(network,entailment)

    def forward(self,q): return q(self.learner)

class MetaLearner(nn.Module):
    pass

class PipelinerLearner(nn.Module):
    def __init__(self,network,entailment):
        super().__init__()

    