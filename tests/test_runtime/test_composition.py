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
    ((1, 3, 980, 1260), np.float32),
    ((1, 3, 980, 1260), np.float16),
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float16),
])
def test_Compose(shape, dtype):
    data = np.ones(shape, dtype)
    #data = np.random.randint(255, size=(280, 280, 3), dtype=np.uint8)
    
    #changing the order will create problems
    ov_preprocess = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        Pad((1, 1)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.Pad((1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ov_result = ov_preprocess(data)[0]
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()
    #print("my_result = ", my_result)
    #print("torch_result = ", torch_result)
    assert np.allclose(ov_result, torch_result, rtol=1e-03) # ieee754


@pytest.mark.parametrize("shape, dtype, p", [
    ((1, 3, 980, 1260), np.float32, 0.5),
    ((1, 3, 980, 1260), np.float16, 0.4),
    ((1, 3, 224, 224), np.float32, 0.8),
    ((1, 3, 224, 224), np.float16, 0.2),
])
@pytest.mark.xfail(reason="since the randomApply has a probability p that the transformation \
                   is applied, the results will not be the same everytime.")
def test_RandomApply(shape, dtype, p):
    data = np.ones(shape, dtype=dtype)

    ov_preprocess = RandomApply([
        Resize((256, 256)),
        Pad((1, 1)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ], p)

    torch_preprocess = transforms.RandomApply([
        transforms.Resize((256, 256)),
        transforms.Pad((1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ], p)

    ov_result = ov_preprocess(data)[0]
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()

    assert np.allclose(ov_result, torch_result, rtol=1e-03)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 980, 1260), np.float32),
    ((1, 3, 224, 224), np.float32),
])
@pytest.mark.xfail(reason="since the randomOrder has a random order of the transformations, \
                   the output will change if the transformation are applied in different order.")
def test_RandomOrder(shape, dtype):
    data = np.ones(shape, dtype)

    np.random.seed(0)  # Set the seed for numpy
    ov_preprocess = RandomOrder([
        Resize((256, 256)),
        Pad((1, 1)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    torch.manual_seed(0)  # Set the seed for PyTorch
    torch_preprocess = transforms.RandomOrder([
        transforms.Resize((256, 256)),
        transforms.Pad((1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ov_result = ov_preprocess(data)[0]
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()

    assert np.allclose(ov_result, torch_result, rtol=1e-03)



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
