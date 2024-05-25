import openvino as ov
import openvino.runtime.opset14 as ops
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