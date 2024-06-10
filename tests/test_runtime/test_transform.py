import pytest
import numpy as np
from torchvision import transforms
import torch
from src.torch_adapter.transforms import *


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.int32),
    ((1, 3, 960, 1280), np.int64),
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 256, 256), np.float64),
])
def test_abs(shape, dtype):
    data = np.random.rand(*shape) + 0.5
    data = data.astype(dtype)
    print("data = ", data)
    preprocess = Compose([
        abs(),
    ])
    tensor = preprocess(data)[0]
    print("result = ", tensor)
    assert np.allclose(tensor, np.abs(data), rtol=1e-03)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.float32),
    ((1, 3, 960, 1280), np.int32),
    ((1, 3, 960, 1280), np.int64),
])
def test_resize(shape, dtype):
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)
    preprocess = Compose([
        Resize(256, 256),
    ])
    tensor = preprocess(data)[0]
    # print("result = ", tensor)
    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
    ])
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()
    # print("torch_result = ", torch_result)
    assert np.allclose(torch_result, tensor, rtol=1e-03)


#TODO
@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float64),
])
@pytest.mark.skip(reason="Need to be fixed")
def test_centerCrop(shape, dtype):
    data = np.random.rand(*shape).astype(dtype)
    preprocess = Compose([
        CenterCrop(112, 112, shape),
    ])
    tensor = preprocess(data)[0]
    torch_preprocess = transforms.Compose([
        transforms.CenterCrop((112, 112)),
    ])
    torch_result = torch_preprocess(data)[0].numpy()
    assert np.allclose(tensor, torch_result, rtol=1e-3)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.int64),
    ((1, 3, 224, 224), np.int32),
])
@pytest.mark.skip(reason="Need to be fixed and to be implemented support for PIL Image")
def test_to_tensor(shape, dtype):
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
    # print("torch_tensor = ", torch_tensor)
    # print("tensor = ", tensor)
    assert tensor is not None
    assert tensor.get(0).shape == (3, 224, 224) if len(shape) == 3 else (1, 224, 224)
    assert tensor.get(0).shape == torch_tensor.shape


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.float32),
    ((1, 3, 960, 1280), np.int32),
    ((1, 3, 960, 1280), np.int64),
])
def test_pad(shape, dtype):
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)
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
    assert np.allclose(tensor, torch_tensor, rtol=1e-03)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 960, 1280), np.float64),
])
def test_normalize(shape, dtype):
    data = np.ones(shape, dtype)
    preprocess = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(data)[0]
    print("tensor = ", tensor)
    torch_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch_tensor = torch_preprocess(torch.tensor(data))[0].numpy()
    print("torch_tensor = ", torch_tensor)
    assert np.allclose(tensor, torch_tensor, rtol=1e-03)
