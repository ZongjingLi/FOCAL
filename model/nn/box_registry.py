import  torch
import  torch.nn as nn

class BoxRegistry(nn.Module):
    _init_methods = {"uniform":torch.nn.init.uniform}

    def __init__(self,config):
        super().__init__()
        self.dim = config.concept_dim
        
        registry_config = config.box_registry
        entries  = registry_config.entries

        init_config = registry_config.INIT
        self.boxes = self._init_embedding_(entries,init_config)
        
        clamp_config = registry_config.offset
        self.offset_clamp = clamp_config.offset
        self.center_clamp = clamp_config.center

    def _init_embedding_(self,entries,init_config):
        return nn.Embedding()

    def forward(self,x):return self.boxes(x)

    def __setitem__(self,key,item):
        self.boxes.weight[key] = item

    def __getitem__(self,key):return self.boxes.weight[key]

    def clamp_dimensions(self):
        with torch.no_grad():
            self.boxes.weight[:,self.dim:].clamp_(*self.offset_clamp)
            self.boxes.weight[:,:self.dim].clamp_(*self.center_clamp)

    @property
    def size(self):return self.dim * 2

class PlaneRegistry(nn.Module):
    def __init__(self,config):
        super().__init__()