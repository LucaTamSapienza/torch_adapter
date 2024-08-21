from src.torch_adapter.transforms import *
from openvino.preprocess import  ColorFormat
import numpy as np
import torch
from torchvision import transforms
import pytest
import urllib


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 240, 240), np.float32),
    ((1, 3, 240, 240), np.float16),
])
def test_Compose_1(shape, dtype):
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)
    
    #TODO: to be fixed -> changing the order will create problems
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

    assert np.allclose(ov_result, torch_result, rtol=1e-02) #TODO: increase the rtol


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 240, 240), np.float16),
    ((1, 3, 240, 240), np.float32),
])
def test_Compose_2(shape, dtype):
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)

    ov_preprocess =  Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        # ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ov_result = ov_preprocess(data)[0]
    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    torch_result = torch_preprocess(torch.tensor(data))[0].numpy()

    assert np.allclose(torch_result, ov_result, rtol=1e-02)


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


"""########################## test PIL images ##########################"""


image_urls = [
    ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"),
    # ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png"),
]

def download_image(url, filename):
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    return Image.open(filename)

input_images = [download_image(url, filename) for url, filename in image_urls]

@pytest.mark.parametrize("input_image", input_images)
def test_compose_pil(input_image):
    ov_preprocess = Compose([
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

    ov_result = ov_preprocess(input_image)[0]
    # print("ov \n", ov_result)
    torch_result = torch_preprocess(input_image)[0].numpy()
    # print("\ntorch \n", torch_result)

    # assert np.allclose(ov_result, torch_result, rtol=1e-02)
