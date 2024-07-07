import torch.hub 
from src.torch_adapter.util import AdapterModel

def load(*args, **kwargs):
    return AdapterModel(torch.hub.load(*args, **kwargs))
