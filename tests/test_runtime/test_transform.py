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
    # print("data = ", data)
    ov_preprocess = Compose([
        abs(),
    ])
    ov_tensor = ov_preprocess(data)[0]
    # print("result = ", ov_tensor)
    assert np.allclose(ov_tensor, np.abs(data), rtol=1e-03)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.float32),
    ((1, 3, 960, 1280), np.int32),
    ((1, 3, 960, 1280), np.int64),
])
def test_resize(shape, dtype):
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)
    ov_preprocess = Compose([
        Resize((256, 256)),
    ])
    ov_tensor = ov_preprocess(data)[0]
    # print("ov_tensor = ", ov_tensor)
    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
    ])
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()
    # print("torch_result = ", torch_result)
    assert np.allclose(torch_result, ov_tensor, rtol=1e-03)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float64),
    ((1, 3, 224, 224), np.int32),
    ((1, 3, 224, 224), np.int64),
])
def test_centerCrop(shape, dtype):
    data = np.random.rand(*shape).astype(dtype)
    preprocess = Compose([
        CenterCrop((112, 112)),
    ])
    ov_result = preprocess(data)[0]
    torch_preprocess = transforms.Compose([
        transforms.CenterCrop((112, 112)),
    ])
    # Convert data to a PyTorch tensor before applying torch_preprocess
    torch_data = torch.tensor(data, dtype = torch.float32) # needs to be dynamic based on the dtype parameter
    torch_result = torch_preprocess(torch_data)[0].numpy()
    # print("torch_result = ", torch_result.shape)
    # print("ov_result = ", ov_result.shape)
    assert np.allclose(ov_result, torch_result, rtol=1e-02)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.int64),
    ((1, 3, 224, 224), np.int32),
])
@pytest.mark.skip(reason="Need to be fixed and to be implemented support for PIL Image")
def test_to_tensor(shape, dtype):
    data = np.random.randint(0, 256, shape, dtype)
    #print(data.ndim)
    ov_preprocess = Compose([
        ToTensor(),
    ])
    torch_preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    ov_tensor = ov_preprocess(data)
    torch_tensor = torch_preprocess(data)
    # print("torch_tensor = ", torch_tensor)
    # print("tensor = ", tensor)
    assert ov_tensor is not None
    assert ov_tensor.get(0).shape == (3, 224, 224) if len(shape) == 3 else (1, 224, 224)
    assert ov_tensor.get(0).shape == torch_tensor.shape


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.float32),
    ((1, 3, 960, 1280), np.int32),
    ((1, 3, 960, 1280), np.int64),
])
def test_pad(shape, dtype):
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)
    ov_preprocess = Compose([
        Pad((1, 1)),
    ])
    ov_tensor = ov_preprocess(data)[0]
    # print("ov_tensor = ", ov_tensor)
    torch_preprocess = transforms.Compose([
        transforms.Pad((1, 1)),
    ])
    torch_tensor = torch_preprocess(torch.tensor(data))[0].numpy()
    # print("torch_tensor = ", torch_tensor)
    assert np.allclose(ov_tensor, torch_tensor, rtol=1e-03)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 960, 1280), np.float64),
])
def test_normalize(shape, dtype):
    data = np.ones(shape, dtype)
    ov_preprocess = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ov_tensor = ov_preprocess(data)[0]
    # print("tensor = ", ov_tensor)
    torch_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch_tensor = torch_preprocess(torch.tensor(data))[0].numpy()
    # print("torch_tensor = ", torch_tensor)
    assert np.allclose(ov_tensor, torch_tensor, rtol=1e-03)

"""
@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.int32),
    ((1, 3, 224, 224), np.int64),
])
def test_convert_image_dtype(shape, dtype):
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)
    ov_preprocess = Compose([
        ConvertImageDtype(torch.float32),
    ])
    ov_tensor = ov_preprocess(data)[0]
    print("ov_tensor = ", ov_tensor)
    torch_preprocess = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
    ])
    torch_tensor = torch_preprocess(torch.tensor(data))[0].numpy()
    print("torch_tensor = ", torch_tensor)
    assert np.allclose(tensor, torch_tensor, rtol=1e-03)
"""
