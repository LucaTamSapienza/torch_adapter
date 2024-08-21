from src.torch_adapter.transforms import create_empty_model
import openvino as ov
import pytest

#first create_empty_model
@pytest.mark.parametrize("input_params", [
    ([2, 8], ov.Type.f32),
    ([4, 4, 3], ov.Type.u8),
    ([1, 224, 224, 3], ov.Type.f32)
])
def test_create_empty_model(input_params):
    with pytest.raises(TypeError):
        create_empty_model(*input_params)
