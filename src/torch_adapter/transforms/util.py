import openvino.runtime.opset14 as ops
import openvino as ov
from openvino.runtime.utils.decorators import custom_preprocess_function


def create_empty_model(shapes, dtype):
    """
    Create an empty input model that can take in different shapes and data types.

    :param shapes: List of shapes
    :param dtype: List of data types
    :return: OpenVINO model
    """
    # If shapes and datatypes have different lengths, fill datatypes with ov.Type.f32
    if len(shapes) != len(dtype):
        dtype.extend([ov.Type.f32] * (len(shapes) - len(dtype)))
    
    parameters = []
    for shape, dtype in zip(shapes, dtype):
        param = ops.parameter(shape, dtype)
        parameters.append(param)

    outputs = []
    for param in parameters:
        output = ops.result(param)
        output.friendly_name = "result"
        outputs.append(output)

    return ov.Model(outputs, parameters, "empty_model")


@custom_preprocess_function
def custom_preprocess_abs(output: ov.runtime.Output):
    return ops.abs(output)