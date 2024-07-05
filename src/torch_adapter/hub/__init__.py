import torch.hub 
from src.torch_adapter.AdapterModel import AdapterModel

def load(*args, **kwargs):
    return AdapterModel(torch.hub.load(*args, **kwargs))
