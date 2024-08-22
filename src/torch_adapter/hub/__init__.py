import torch.hub 
from ..util import AdapterModel

def load(*args, **kwargs):
    return AdapterModel(torch.hub.load(*args, **kwargs))
