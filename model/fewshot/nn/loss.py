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
        for i,categories in enumerate(inputs["val_sample"]["category"]):
            ends = outputs["val_sample"]["end"][i]

            targets = inputs["val_sample"]["target"][i]
            outputs = inputs["val_sample"]["answer_tokenized"][i]


            ls = []

            for j,category in enumerate(categories):
                pass

            losses.append(torch.stack(ls).mean())

        return {"validation_loss":torch.stack(losses).mean()}