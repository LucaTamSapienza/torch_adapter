from src.torch_adapter.AdapterModel import AdapterModel
from src.torch_adapter.transforms import *
import torch
import pytest
from torchvision import transforms
import numpy as np


@pytest.mark.parametrize("input_shape", [(1, 3, 224, 224), (1, 3, 299, 299)])
def test_inference(input_shape):
    adapter_model = AdapterModel(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True))
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
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    data = np.random.rand(*shape) + 0.5
    data = data.astype(dtype)

    ov_model = AdapterModel(model)

    model.eval()

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
        output = model(torch_tensor)
    
    #print(output[0])
    
    ov_model.eval()
    ov_preprocess = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        # ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ov_tensor = ov_preprocess(data)
    ov_output = ov_model(ov_tensor)

    #print(ov_output[0])

    assert np.allclose(ov_output[0], output[0].numpy(), rtol=1e-01)
