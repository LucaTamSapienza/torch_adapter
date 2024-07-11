import pytest
import torch
import torch.hub
import numpy as np
from torchvision import transforms
import src.torch_adapter as torch_adapter
from src.torch_adapter.transforms import Compose, Resize, CenterCrop, Normalize
import timeit

clock = 200

@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float16),
])
def test_inference_with_transforms(shape, dtype):
    torch_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    ov_model = torch_adapter.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    
    data = np.random.rand(*shape) + 1.0
    data = data.astype(dtype)

    torch_model.eval()
    torch_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    torch_tensor = torch_preprocess(torch.tensor(data, dtype=torch.float32))

    # Inference and benchmarking for PyTorch
    def torch_inference():
        with torch.no_grad():
            output = torch_model(torch_tensor)
        return output

    torch_time = timeit.timeit(torch_inference, number=clock)
    print(f"PyTorch Inference Time: {torch_time / clock:.6f} seconds per inference")

    ov_model.eval()
    ov_preprocess = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ov_tensor = ov_preprocess(data)

    # Inference and benchmarking for OpenVINO
    def ov_inference():
        ov_output = ov_model(ov_tensor)[0]
        return ov_output

    ov_time = timeit.timeit(ov_inference, number=clock)
    print(f"OpenVINO Inference Time: {ov_time / clock:.6f} seconds per inference")
