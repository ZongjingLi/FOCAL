import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device",     default = device)

# build the concept engine
parser.add_argument("--concept_dim",default = 100)
parser.add_argument("concept_repr",default = "box")

# add details in the construction loss
parser.add_argument("--fewshot_tau",default = 0.2)
parser.add_argument("--temperature",default = 0.2)

# add the training epoch details 
parser.add_argument("--epoch",   default = 1000)
parser.add_argument("--lr",      default = 2e-3)
parser.add_argument("--batch",   default = 4)
parser.add_argument("--shuffle", default = True)
config = parser.parse_args(args = [])