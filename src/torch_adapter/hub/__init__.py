import torch.hub 
from src.torch_adapter import AdapterModel as Model

def load(*args, **kwargs):
    return Model(torch.hub.load(*args, **kwargs))