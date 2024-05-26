import openvino.runtime.opset14 as ops
import openvino as ov
from openvino.runtime.utils.decorators import custom_preprocess_function

@custom_preprocess_function
def custom_preprocess_abs(output: ov.runtime.Output):
    return ops.abs(output)