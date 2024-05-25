from src.torch_adapter.transforms import *
from openvino.preprocess import  ColorFormat
import numpy as np
import torch
from torchvision import transforms
import pytest

#import openvino as ov
"""
# Defining the preprocessing tranformations
preprocess = Compose([
    # Resize(256, 256),
    # ToTensor()
    # ConvertColor(ColorFormat.RGB),
    Scale(2.0),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Creating preprocessing model
data = np.ones((1, 3, 224, 224), dtype=np.float32)
print("data = ", data)

# preprocessing...
preprocess_model = preprocess([data.shape], [data.dtype])
print("preprocess_model = ", preprocess_model)
result = preprocess_model(data)[0]
print("result = ", result)
"""

@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float64),
])
def test_Compose(shape, dtype):
    data = np.ones(shape, dtype=dtype)

    my_preprocess = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    torch_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    my_result = my_preprocess([data.shape], [data.dtype])(data)[0]
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()

    assert np.allclose(my_result, torch_result, rtol=1e-03)
