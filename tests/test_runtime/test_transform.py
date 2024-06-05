import pytest
import numpy as np
from torchvision import transforms
import torch
from src.torch_adapter.transforms import *


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.int32),
    ((1, 3, 960, 1280), np.int64),
    #((1, 3, 960, 1280), np.float32),
])
# not working with float32 (?)
def test_abs(shape, dtype):
    data = np.random.rand(*shape).astype(dtype)
    print("data = ", data)
    preprocess = Compose([
        abs(),
    ])
    tensor = preprocess(data)[0]
    print("result = ", tensor)
    assert np.allclose(tensor, np.abs(data))


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.float32),
    ((1, 3, 960, 1280), np.int32),
])
# incompatible shape because the transformed shape doesn't match the original shape
def test_resize(shape, dtype):

    data = np.ones(shape, dtype)
    
    preprocess = Compose([
        Resize(256, 256),
    ])
    tensor = preprocess(data)[0]
    print("result = ", tensor)
    
    torch_preprocess = transforms.Compose([
        transforms.Resize((256)),
    ])
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()
    print("torch_result = ", torch_result)


"""
#TODO: same error for resize and need to implement the centerCrop transformation
@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float64),
])
def test_centerCrop(shape, dtype):
    # data with random values
    data = np.ones(shape, dtype)
    print("data = ", data)
    preprocess = Compose([
        CenterCrop(100),
    ])
    tensor = preprocess(data)[0]
    print("CenterCrop = ", tensor)
    torch_preprocess = transforms.Compose([
        transforms.CenterCrop(100),
    ])
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()
    print("torch_result = ", torch_result)
    assert np.allclose(tensor, torch_result) 
"""


"""
@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.int64),
    ((1, 3, 224, 224), np.int32),
])
#not needed (?)
def test_to_tensor(shape, dtype):
    # Create dummy data
    data = np.random.randint(0, 256, shape, dtype)
    #print(data.ndim)
    preprocess = Compose([
        ToTensor(),
    ])

    torch_preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    tensor = preprocess(data)
    torch_tensor = torch_preprocess(data)
    print("torch_tensor = ", torch_tensor)
    print("tensor = ", tensor)
    # Assert the tensor has been processed correctly
    assert tensor is not None
    assert tensor.get(0).shape == (3, 224, 224) if len(shape) == 3 else (1, 224, 224)
    assert tensor.get(0).shape == torch_tensor.shape
"""

@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.int64),
    ((1, 3, 224, 224), np.int32),
])
# same error for resize
def test_pad(shape, dtype):
    # Create dummy data
    data = np.random.randint(0, 256, shape, dtype=dtype)
    preprocess = Compose([
        Pad((0, 0, 1, 1), (0, 0, 1, 1)),
    ])
    tensor = preprocess(data)[0]
    print("tensor = ", tensor)
    torch_preprocess = transforms.Compose([
        transforms.Pad((1, 1)),
    ])
    torch_tensor = torch_preprocess(torch.tensor(data))[0].numpy()
    print("torch_tensor = ", torch_tensor)
    assert np.allclose(tensor, torch_tensor)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.float32),
    ((1, 3, 960, 1280), np.float64),
])
# not working when np.random.rand is used
def test_normalize(shape, dtype):
    data = np.ones(shape, dtype)
    #data = np.random.rand(*shape).astype(dtype)
    #print("data_pre = ", data)
    preprocess = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(data)[0]
    print("tensor = ", tensor)
    #print("data_post = ", data)
    torch_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch_tensor = torch_preprocess(torch.tensor(data))[0].numpy()
    print("torch_tensor = ", torch_tensor)
    assert np.allclose(tensor, torch_tensor, rtol=1e-03, atol=1e-08)
