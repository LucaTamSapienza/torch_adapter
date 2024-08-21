import src.torch_adapter as torch_adapter
from src.torch_adapter.transforms import *
import torch
import pytest
from torchvision import transforms
import numpy as np
from tests import util as f


@pytest.mark.parametrize("input_shape", [(1, 3, 224, 224), (1, 3, 299, 299)])
def test_inference(input_shape):
    adapter_model = torch_adapter.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    adapter_model.eval()
    input_tensor = torch.randn(input_shape)
    output = adapter_model(input_tensor)
    assert output.get(0).shape == (1, 1000)


# TODO: profile
@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float16),
])
def test_inference_with_transforms(shape, dtype):
    torch_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    ov_model = torch_adapter.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) # AdapterModel
    # data = f.create_data(shape, dtype)
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)

    torch_model.eval()

    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    torch_tensor = torch_preprocess(torch.tensor(data, dtype=torch.float32))
    # input_batch = torch_tensor.unsqueeze(0)

    with torch.no_grad():
        output = torch_model(torch_tensor)
 
    ov_model.eval()
    ov_preprocess = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        # ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ov_tensor = ov_preprocess(data)
    ov_output = ov_model(ov_tensor)[0]

    # assert np.allclose(ov_tensor[0], torch_tensor[0].numpy(), rtol=1e-02) # True

    # print(np.sort(ov_output - output[0].numpy())) # print in ascending order the difference between the two arrays

    assert np.argmax(ov_output) == np.argmax(output[0].numpy())

    # f.print_close_broken_elements(output[0].numpy(), ov_output)
