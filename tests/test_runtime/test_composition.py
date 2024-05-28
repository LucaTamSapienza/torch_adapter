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
        # Pad((1, 1), (1, 1), "constant", 0.0),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    torch_preprocess = transforms.Compose([
        # transforms.Pad((1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    my_result = my_preprocess(data)[0]
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()

    assert np.allclose(my_result, torch_result, rtol=1e-03)


@pytest.mark.parametrize("shape, dtype, p", [
    ((1, 3, 224, 224), np.float32, 1.0),
    ((1, 3, 224, 224), np.float64, 0.5),
])
def test_RandomApply(shape, dtype, p):
    data = np.ones(shape, dtype=dtype)

    my_preprocess = RandomApply([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ], p)

    torch_preprocess = transforms.RandomApply([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ], p)

    my_result = my_preprocess(data)[0]
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()

    assert np.allclose(my_result, torch_result, rtol=1e-03)

# TODO: Needs to be tested with more transformations
@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float64),
])
def test_RandomOrder(shape, dtype):
    data = np.ones(shape, dtype=dtype)

    np.random.seed(0)  # Set the seed for numpy
    my_preprocess = RandomOrder([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    torch.manual_seed(0)  # Set the seed for PyTorch
    torch_preprocess = transforms.RandomOrder([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    my_result = my_preprocess(data)[0]
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()

    assert np.allclose(my_result, torch_result, rtol=1e-03)

"""
import urllib
from PIL import Image

@pytest.mark.parametrize("url, filename", [
    ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"),
    # Add more tuples here for additional test cases
])
def test_Compose(url, filename):
    # Download the image
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    # Open the image with PIL
    input_image = Image.open(filename)

    # Define the preprocessing transformations
    my_preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    torch_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    my_result = my_preprocess(input_image)
    torch_result = torch_preprocess(input_image)

    # Convert the results to numpy arrays
    my_result = np.array(my_result)
    torch_result = np.array(torch_result)

    # Check if the results are close
    assert np.allclose(my_result, torch_result, rtol=1e-03)
"""