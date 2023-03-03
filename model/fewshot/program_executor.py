import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nn import build_entailment
from utils import underscores, freeze
