import src.torch_adapter.transforms.util as f
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
from openvino.runtime import Layout


# Base class for all transformations
class Transform:
    def __call__(self, ppp):
        raise NotImplementedError("Transformations must implement the __call__ method.")


# Scale transformation
class Scale(Transform):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, ppp):
        ppp.input().preprocess().scale(self.factor)


# Resize transformation
class Resize(Transform):
    def __init__(self, height, width=None):
        self.height = height
        # If width is not provided, set it to height
        self.width = width if width is not None else height

    def __call__(self, ppp):
        ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR, self.height, self.width)

# TODO: Implement CenterCrop transformation
class CenterCrop(Transform):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, ppp):
        pass


# Normalize transformation
class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, ppp):
        ppp.input().preprocess().mean(self.mean).scale(self.std)


# Color conversion transformation
class ConvertColor(Transform):
    def __init__(self, color_format):
        self.color_format = color_format

    def __call__(self, ppp):
        ppp.input().preprocess().convert_color(self.color_format)


# TODO: Implement ToTensor transformation
class ToTensor(Transform):
    def __init__(self):
        pass
    def __call__(self, ppp):
        pass


# TODO: Implement ToTensor transformation
class ConvertImageDtype(Transform):
    def __init__(self):
        pass
    def __call__(self, ppp):
        pass


# TODO: Implement ToTensor transformation
class CenterCrop(Transform):
    def __init__(self):
        pass
    def __call__(self, ppp):
        pass

"""
# Pad transformation
def custom_pad_factory(pads_begin, pads_end, pad_mode, arg_pad_value):
    @custom_preprocess_function
    def custom_pad(output: ov.runtime.Output):
        pads_begin_node = ops.constant(np.array(pads_begin, dtype=np.int64))
        pads_end_node = ops.constant(np.array(pads_end, dtype=np.int64))
        if arg_pad_value is not None:
            pad_value_node = ops.constant(np.array(arg_pad_value, dtype=map_dtype(output.get_element_type())))
            return ops.pad(output, pads_begin_node, pads_end_node, pad_value_node, pad_mode)
        else:
            return ops.pad(output, pads_begin_node, pads_end_node, pad_mode)
    return custom_pad

class Pad(Transform):
    def __init__(self, pads_begin, pads_end, pad_mode="constant", arg_pad_value=None):
        self.pads_begin = pads_begin
        self.pads_end = pads_end
        self.pad_mode = pad_mode
        self.arg_pad_value = arg_pad_value

    def __call__(self, ppp):
        pad_func = custom_pad_factory(self.pads_begin, self.pads_end, self.pad_mode, self.arg_pad_value)
        ppp.output(0).postprocess().custom(pad_func)
"""


# trying abs preprocessing
class abs(Transform):
    def __init__(self):
        pass
    def __call__(self, ppp):
        return ppp.output(0).postprocess().custom(f.custom_preprocess_abs)