import src.torch_adapter as torch_adapter
from src.torch_adapter.AdapterModel import AdapterModel
from src.torch_adapter.transforms import *
import torch
import pytest
from torchvision import transforms
import numpy as np


@pytest.mark.parametrize("input_shape", [(1, 3, 224, 224), (1, 3, 299, 299)])
def test_inference(input_shape):
    adapter_model = torch_adapter.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    adapter_model.eval()
    input_tensor = torch.randn(input_shape)
    output = adapter_model(input_tensor)
    assert output.get(0).shape == (1, 1000)


@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float16),
    #((1, 3, 224, 224), np.int32),
    #((1, 3, 224, 224), np.int64),
])
def test_inference_with_transforms(shape, dtype):
    torch_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    ov_model = torch_adapter.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    data = np.random.rand(*shape).astype(dtype)


    torch_model.eval()

    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    torch_data = torch.tensor(data, dtype = torch.float32)
    torch_tensor = torch_preprocess(torch_data)
    # input_batch = torch_tensor.unsqueeze(0)


    with torch.no_grad():
        output = torch_model(torch_tensor)
    
    #print(output[0])
    
    ov_model.eval()
    ov_preprocess = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        # ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ov_tensor = ov_preprocess(data)
    ov_output = ov_model(ov_tensor)[0]

    #raise RuntimeError

    #print(ov_output[0])

    print(np.sort(ov_output - output[0].numpy()))

    assert np.allclose(ov_output, output[0].numpy(), rtol=1e-01)


    """
    close_elements = np.isclose(output, ov_output, rtol=1e-05, atol=1e-08)
    # Use np.sum to count the number of True values in the boolean array
    count_close_elements = np.sum(close_elements)
    # Print the number of close elements
    print(f'Number of close elements: {count_close_elements}')
    # To get the number of "broken" or not close elements, subtract the count of close elements from the total number of elements
    count_broken_elements = output.size - count_close_elements
    # Print the number of broken elements
    print(f'Number of broken elements: {count_broken_elements}')
    """

