import openvino.runtime.opset14 as ops
import openvino as ov
from openvino.runtime import Tensor
from openvino.runtime.utils.decorators import custom_preprocess_function
from typing import List
import copy
from typing import Sequence as SequenceType
from collections.abc import Sequence
import numbers
from typing import Any, Tuple
from functools import singledispatch


def create_empty_model(shapes, dtype):
    """
    Create an empty input model that can take in different shapes and data types.

    :param shapes: List of shapes
    :param dtype: List of data types
    :return: OpenVINO model
    """
    
    # the model needs to be "dynamic" so we need to fill the shapes with -1
    shapes = [[-1 for _ in range(len(shape))] for shape in shapes] 

    # If shapes and datatypes have different lengths, fill datatypes with ov.Type.f32
    if len(shapes) != len(dtype):
        dtype.extend([ov.Type.f32] * (len(shapes) - len(dtype)))
    
    parameters = [ops.parameter(shape, dtype) for shape, dtype in zip(shapes, dtype)]

    def set_friendly_name(output):
        output.friendly_name = "result"
        return output

    outputs = [set_friendly_name(ops.result(param)) for param in parameters]

    return ov.Model(outputs, parameters, "empty_model")


@custom_preprocess_function
def custom_preprocess_abs(output: ov.runtime.Output):
    return ops.abs(output)


def _NHWC_to_NCHW(shape) -> Tuple:
    return (shape[0], shape[3], shape[1], shape[2])

@singledispatch
def _setup_size(size: Any, error_msg: str) -> SequenceType[int]:
    raise ValueError(error_msg)


@_setup_size.register
def _setup_size_number(size: numbers.Number, error_msg: str) -> SequenceType[int]:
    return int(size), int(size)  # type: ignore


@_setup_size.register
def _setup_size_sequence(size: Sequence, error_msg: str) -> SequenceType[int]:
    if len(size) == 1:
        return size[0], size[0]
    elif len(size) == 2:
        return size[0], size[1]
    raise ValueError(error_msg)
