import torch

device = "cuda:0" if torch.cuda.is_availabel() else "cpu"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--concept_dim",default = 100)

config = parser.parse_args(args = [])