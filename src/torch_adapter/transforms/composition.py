import numpy as np
from openvino.preprocess import PrePostProcessor
import openvino as ov
import openvino.runtime.opset14 as ops
from openvino.runtime import Layout


# TODO: __call__ method should not take shape as an argument
def create_empty_model(shapes, dtype):
    """
    Create an empty input model that can take in different shapes and data types.

    :param input_specs: List of tuples where each tuple contains (shape, datatype)
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


# Compose class is used to compose multiple transforms together.
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, shape, dtype):
        
        model = create_empty_model(shape, dtype)
        ppp = PrePostProcessor(model)

        # Set the layout of the input tensor to NCHW by default
        ppp.input().tensor().set_layout(Layout("NCHW"))

        for transform in self.transforms:
            transform(ppp)

        final_model = ppp.build()
        return ov.Core().compile_model(final_model, "CPU")
    
# RandomApply class is used to apply a list of transforms randomly with a given probability p (0.5 by default)
class RandomApply:
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, shape, dtype):
        model = create_empty_model(shape, dtype)
        ppp = PrePostProcessor(model)

        ppp.input().tensor().set_layout(Layout("NCHW"))
        if np.random.random() < self.p:
            for transform in self.transforms:
                transform(ppp)
        ppp.build()
        return ov.Core().compile_model(model, "CPU")
    
# RandomOrder class is used to apply a list of transforms in a random order
class RandomOrder:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, shape, dtype):

        model = create_empty_model(shape, dtype)
        ppp = PrePostProcessor(model)
        ppp.input().tensor().set_layout(Layout("NCHW"))
        order = np.random.permutation(len(self.transforms))

        for i in order:
            self.transforms[i](ppp)

        ppp.build()
        return ov.Core().compile_model(model, "CPU")
