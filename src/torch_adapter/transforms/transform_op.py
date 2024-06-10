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
    
    result: resized Image
    """
    def __init__(self, height, width, algorithm=ResizeAlgorithm.RESIZE_BILINEAR_PILLOW):
        self.height = height
        self.width = width
        self.algorithm = algorithm

    def __call__(self, ppp):
        input_data = ppp.input()
        # print("input_data = ", input_data)
        ppp.input().preprocess().resize(self.algorithm, self.height, self.width)



# CenterCrop transformation
#TODO
class CenterCrop(Transform):
    def __init__(self, crop_height, crop_width, shape):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.shape = shape

    def __call__(self, ppp):
        # Recupera la dimensione dell'immagine di input
        input_height, input_width = self.shape[-2], self.shape[-1]

        # Calcola le coordinate superiori e sinistre per il ritaglio centrato
        crop_top = (input_height - self.crop_height) // 2
        crop_left = (input_width - self.crop_width) // 2

        # Define the top-left and bottom-right coordinates of the crop
        crop_top_left = [crop_top, crop_left]
        crop_bottom_right = [crop_top + self.crop_height, crop_left + self.crop_width]

        # Applica il ritaglio centrato
        ppp.input().preprocess().crop(crop_top_left, crop_bottom_right)
        

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


# ToTensor transformation
# need to implement support for Pil image
class ToTensor(Transform):
    """
    convert numpy.ndarray -> Tensor
    TBA support PIL Image
    """
    def __init__(self):
        pass
    def __call__(self, ppp):
        print("ppp = ", ppp.__class__.__name__)
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