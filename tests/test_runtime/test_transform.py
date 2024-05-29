import pytest
import numpy as np
from torchvision import transforms
import torch
from src.torch_adapter.transforms import *
"""
@pytest.mark.parametrize("pads_begin, pads_end", [((1, 1), (1, 1)), ((2, 2), (2, 2)), ((1, 2), (2, 1))])
def test_pad_op(pads_begin, pads_end):
    # Create an instance of the Pad class
    pad = Pad(pads_begin, pads_end)

    # Create some input data
    input_data = np.ones((3, 3))
    input_tensor = torch.from_numpy(input_data)

    # Apply custom Pad operation
    output_data = pad(input_tensor)

    # Apply PyTorch's pad function
    expected_output_data = torch.nn.functional.pad(input_tensor, (pads_begin[0], pads_end[0], pads_begin[1], pads_end[1]))

    # Check if the output data is as expected
    assert torch.allclose(output_data, expected_output_data)
"""

# print with pytest --> pytest path_to_test_runtime/test_transform.py -s 
@pytest.mark.parametrize("shape, dtype", [
    ((1, 3, 224, 224), np.float32),
    ((1, 3, 224, 224), np.float64),
])
def test_abs(shape, dtype):
    data = np.ones(shape, dtype) * -1
    print("data = ", data)
    preprocess = Compose([
        abs(),
    ])
    tensor = preprocess(data)[0]
    print("result = ", tensor)

