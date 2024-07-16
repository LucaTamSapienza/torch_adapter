import pytest
from pytest import param
import numpy as np
from torchvision import transforms
import torch
from src.torch_adapter.transforms import *
from tests import util as f


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 960, 1280), np.int32),
    ((1, 3, 960, 1280), np.int64),
    ((1, 3, 1000, 1000), np.float16),
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 256, 256), np.float64),
])
def test_abs(shape, dtype):
    data = f.create_data(shape, dtype)
    # print("data = ", data)
    ov_preprocess = Compose([
        abs(),
    ])
    ov_tensor = ov_preprocess(data)[0]
    # print("result = ", ov_tensor)
    assert np.allclose(ov_tensor, np.abs(data), rtol=1e-03)


#TODO: Solve problem when type is int / uint
@pytest.mark.parametrize("shape, dtype, interpolation", [
    param((1, 3, 224, 224), np.float32, transforms.InterpolationMode.BILINEAR),
    param((1, 3, 200, 240), np.float32, transforms.InterpolationMode.BILINEAR),
    param((1, 3, 224, 224), np.float32, transforms.InterpolationMode.NEAREST, marks=pytest.mark.xfail(reason="Interpolation not BILINEAR")), # Not Working (?)
    param((1, 3, 200, 240), np.float32, transforms.InterpolationMode.BICUBIC, marks=pytest.mark.xfail(reason="Interpolation not BILINEAR")), # sometimes work, sometimes not
], ids=lambda interpolation: f"interpolation={interpolation}")

def test_resize(shape, dtype, interpolation):
    data = f.create_data(shape, dtype)
    ov_preprocess = Compose([
        Resize((256, 256), interpolation),
    ])
    ov_tensor = ov_preprocess(data)[0]
    # print("ov_tensor = ", ov_tensor)
    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256), interpolation),
    ])
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()

    f.print_close_broken_elements(torch_result, ov_tensor)

    # print("torch_result = ", torch_result)
    assert np.allclose(torch_result, ov_tensor, rtol=1e-02)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float64),
    ((1, 3, 224, 224), np.int32),
    ((1, 3, 224, 224), np.int64),
])
def test_centerCrop(shape, dtype):
    data = f.create_data(shape, dtype)
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


@pytest.mark.parametrize("test_input, pad, fill, padding_mode", [
    (np.random.randint(255, size = (1, 3, 960, 1280), dtype = np.uint8), (1, 1), 0, "constant"),
    (np.random.randint(255, size = (1, 3, 220, 224), dtype = np.uint8), (2), 0, "constant"),
    (np.random.randint(255, size = (1, 3, 960, 1280), dtype = np.uint8), (2, 3), 0, "constant"),
    (np.random.randint(255, size = (1, 3, 960, 1280), dtype = np.uint8), (2, 3, 4, 5), 0, "constant"),
    (np.random.randint(255, size = (1, 3, 960, 1280), dtype = np.uint8), (2, 3, 4, 5), 3, "constant"),
    (np.random.randint(255, size = (1, 3, 218, 220), dtype = np.uint8), (2, 3), 0, "edge"),
    (np.random.randint(255, size = (1, 3, 218, 220), dtype = np.uint8), (2, 3), 0, "reflect"),
    (np.random.randint(255, size = (1, 3, 218, 220), dtype = np.uint8), (2, 3), 0, "symmetric"),
])
def test_pad(test_input, pad, fill, padding_mode):
    ov_preprocess = Compose([
        Pad(pad, fill=fill, padding_mode=padding_mode),
    ])
    ov_tensor = ov_preprocess(test_input)[0]
    # print("ov_tensor = ", ov_tensor)
    torch_preprocess = transforms.Compose([
        transforms.Pad(pad, fill=fill, padding_mode=padding_mode),
    ])
    torch_tensor = torch_preprocess(torch.tensor(test_input))[0].numpy()
    # print("torch_tensor = ", torch_tensor)
    assert np.allclose(ov_tensor, torch_tensor, rtol=4e-05)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 1000, 1000), np.float64),
])
def test_normalize(shape, dtype):
    data = f.create_data(shape, dtype)
    ov_preprocess = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ov_tensor = ov_preprocess(data)[0]
    # print("tensor = ", ov_tensor)
    torch_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch_tensor = torch_preprocess(torch.tensor(data))[0].numpy()
    # print("torch_tensor = ", torch_tensor)
    f.print_close_broken_elements(torch_tensor, ov_tensor)
    assert np.allclose(ov_tensor, torch_tensor, rtol=1e-03)


@pytest.mark.parametrize("shape, dtype, rtol", [
    ((1, 3, 224, 224), np.float16, 1e-05),
    ((1, 3, 224, 224), np.float32, 2e-03),
    ((1, 3, 224, 224), np.uint8, 1e-05),
    ((1, 3, 224, 224), np.int32, 1e-05),
    ((1, 3, 224, 224), np.int64, 1e-05),
])
def test_convert_image_dtype(shape, dtype, rtol):
    data = np.random.rand(*shape).astype(dtype) # problem if using f.create_data(shape, dtype)
    ov_preprocess = Compose([
        ConvertImageDtype(torch.float16),
        ConvertImageDtype(torch.float32),
    ])
    ov_tensor = ov_preprocess(data)[0]
    # print("ov_tensor = ", ov_tensor)
    torch_preprocess = transforms.Compose([
        transforms.ConvertImageDtype(torch.float16),
        transforms.ConvertImageDtype(torch.float32),
    ])
    torch_tensor = torch_preprocess(torch.tensor(data))[0].numpy()
    # print("torch_tensor = ", torch_tensor)
    assert np.allclose(ov_tensor, torch_tensor, rtol=rtol)

