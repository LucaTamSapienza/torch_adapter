from . import util as f
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
from openvino.runtime import Layout
import openvino.runtime.opset14 as ops
import numpy as np
from typing import Union, List, Optional
import openvino as ov
from openvino.runtime.utils.decorators import custom_preprocess_function



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
    """
    input: PIL image or Tensor
    """
    def __init__(self, height, width, algorithm=ResizeAlgorithm.RESIZE_LINEAR):
        self.height = height
        self.width = width
        self.algorithm = algorithm

    def __call__(self, ppp):
        input_data = ppp.input()
        # Get the shape of the input tensor (HOW??????)
        #input_shape = input_data.get_shape()
        # Set the shape of the tensor to the resized dimensions
        print("input_data = ", input_data)
        #ppp.input().tensor().set_shape([input_shape[0], input_shape[1], self.height, self.width])
        ppp.input().preprocess().resize(self.algorithm, self.height, self.width)



# CenterCrop transformation
#TODO
class CenterCrop(Transform):
    def __init__(self, height_ratio, width_ratio):
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def __call__(self, ppp):
        # Get the shape of the input tensor
        #HOW

        # Calculate the crop dimensions
        crop_height = int(input_height * self.height_ratio)
        crop_width = int(input_width * self.width_ratio)

        # Calculate the coordinates for the center crop
        y1 = (input_height - crop_height) // 2
        y2 = y1 + crop_height
        x1 = (input_width - crop_width) // 2
        x2 = x1 + crop_width

        # Apply the crop
        ppp.input().preprocess().crop([x1, y1], [x2, y2])

# Normalize transformation
class Normalize(Transform):
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        (TBA: inplace(bool,optional): Bool to make this operation inplace.)

    Returns:
        Tensor: Normalized Tensor image.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, ppp):
        # input_data = ppp.input().tensor()
        # print("input_data = ", input_data)
        # print("Methods and attributes: ", dir(input_data)) # to see methods avaible
        ppp.input().preprocess().mean(self.mean).scale(self.std)


# Color conversion transformation
class ConvertColor(Transform):
    def __init__(self, color_format):
        self.color_format = color_format

    def __call__(self, ppp):
        ppp.input().preprocess().convert_color(self.color_format)


# Not needed (?)
class ToTensor(Transform):
    """
    convert PIL image or numpy.ndarray -> Tensor
    """
    def __init__(self):
        pass
    def __call__(self, ppp):
        ppp.input().tensor()


# TODO: Implement ConvertImageDyype transformation
class ConvertImageDtype(Transform):
    """
    Convert a tensor image to the given ``dtype`` and scale the values accordingly
    This function does not support PIL Image.

    Args:
        image (Tensor): Image to be converted
        dtype (dtype): Desired data type of the output

    Returns:
        Tensor: Converted image

    """
    def __init__(self):
        pass
    def __call__(self, ppp):
        pass


# Pad transformation
#TODO
class Pad(Transform):
    def __init__(self, padding, fill, mode = "constant"):
        self.padding = padding
        self.fill = fill
        self.mode = mode

    def custom_preprocess_pad(self):
        @custom_preprocess_function
        def inner(output: ov.runtime.Output):
            return ops.pad(output, self.padding, self.fill, self.mode)
        return inner

    def __call__(self, ppp):
        ppp.input().preprocess().custom(self.custom_preprocess_pad())




# trying custom preprocessing
class abs(Transform):
    def __init__(self):
        pass
    def __call__(self, ppp):
        return ppp.input().preprocess().custom(f.custom_preprocess_abs)