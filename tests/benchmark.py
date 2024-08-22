import pytest
import torch
import torch.hub
import numpy as np
from torchvision import transforms
import src.torch_adapter as torch_adapter
from src.torch_adapter.transforms import Compose, Resize, CenterCrop, Normalize
import timeit

clock = 100

def benchmark_inference():
    torch_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    ov_model = torch_adapter.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    
    data = np.ones((8, 3, 1000, 1000), dtype=np.float32)

    torch_model.eval()
    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    torch_tensor = torch_preprocess(torch.tensor(data, dtype=torch.float32))

    # Inference and benchmarking for PyTorch
    def torch_inference():
        with torch.no_grad():
            output = torch_model(torch_tensor)
        return output

    torch_time = timeit.timeit(torch_inference, number=clock)
    print(f"PyTorch Inference Time: {torch_time / clock:.6f} seconds per inference")

    ov_model.eval()
    ov_preprocess = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ov_tensor = ov_preprocess(data)

    # Inference and benchmarking for OpenVINO
    def ov_inference():
        ov_output = ov_model(ov_tensor)[0]
        return ov_output

    ov_time = timeit.timeit(ov_inference, number=clock)
    print(f"OpenVINO Inference Time: {ov_time / clock:.6f} seconds per inference")

benchmark_inference()


"""   
# start ipython, copy and paste the following code (choose which transformation to test) 

import torch
import torch.hub
import numpy as np
from torchvision import transforms
import src.torch_adapter as torch_adapter
from src.torch_adapter.transforms import *
import timeit
import urllib
from PIL import Image

data = np.ones((8, 3, 1000, 1000), dtype=np.float32)

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)
# data = Image.open(filename)

def get_torch_Compose():
    return transforms.Compose([
                #transforms.Resize(256),
                #transforms.CenterCrop(224),
                #transforms.Pad((1, 1)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                #transforms.ConvertImageDtype(torch.float16),
                #transforms.ConvertImageDtype(torch.float32),
        ])


def torch_Compose(data):
    torch_preprocess = get_torch_Compose()

    for i in range(10):
        torch_result = torch_preprocess(torch.from_numpy(data))[0].numpy()

    return torch_result

def get_ov_Compose():
    return Compose([
                #Resize(256),
                #CenterCrop(224),
                #Pad((1, 1)),
                # ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                #ConvertImageDtype(torch.float16),
                #ConvertImageDtype(torch.float32),
        ])

# @profile
def ov_Compose(data):
    my_preprocess = get_ov_Compose()  # That itself was a reason that recompilation always occurred

    for i in range(10):  # if you run a loop that shows the hits to recompile were present
        my_result = my_preprocess(data)[0]

    return my_result

torch_preprocess = get_torch_Compose()
my_preprocess = get_ov_Compose()


# and run it using

%timeit _ = torch_preprocess(torch.from_numpy(data))[0].numpy()) # for torch_transforms
%timeit _ = my_preprocess(data)[0] # for ov_transforms


"""