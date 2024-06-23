from . import util as f
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
from openvino.runtime import Layout, Type
import openvino.runtime.opset14 as ops
import numpy as np
from typing import Tuple, List, Callable, Dict
import openvino as ov
from openvino.runtime.utils.decorators import custom_preprocess_function
import torch
from torchvision.transforms import InterpolationMode

TORCHTYPE_TO_OVTYPE = {
    float: ov.Type.f32,
    int: ov.Type.i32,
    bool: ov.Type.boolean,
    torch.float16: ov.Type.f16,
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.uint8: ov.Type.u8,
    torch.int8: ov.Type.i8,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
    torch.bool: ov.Type.boolean,
    torch.DoubleTensor: ov.Type.f64,
    torch.FloatTensor: ov.Type.f32,
    torch.IntTensor: ov.Type.i32,
    torch.LongTensor: ov.Type.i64,
    torch.BoolTensor: ov.Type.boolean,
}


# Base class for all transformations
class Transform:
    def __call__(self, ppp):
        raise NotImplementedError("Transformations must implement the __call__ method.")


# Scale transformation
#TODO: to be tested
class Scale(Transform):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        ppp.input().preprocess().scale(self.factor)
        return [meta["input_shape"], meta["layout"]]


# Resize transformation
class Resize(Transform):
    def __init__(self, size: Tuple[int, int], interpolation=InterpolationMode.BILINEAR, max_size = 0):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
    
    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        resize_mode_map = {
            InterpolationMode.NEAREST: ResizeAlgorithm.RESIZE_NEAREST,
            InterpolationMode.BILINEAR: ResizeAlgorithm.RESIZE_BILINEAR_PILLOW,
            InterpolationMode.BICUBIC: ResizeAlgorithm.RESIZE_BICUBIC_PILLOW,
        }
        if self.max_size:
            raise ValueError("Resize with max_size if not supported")
        if self.interpolation not in resize_mode_map.keys():
            raise ValueError(f"Interpolation mode {self.interpolation} is not supported.")

        target_h, target_w = self.size

        # rescale the smaller image edge
        current_h, current_w = meta["input_shape"][2:4] if meta["layout"] == Layout("NCHW") else meta["input_shape"][1:3]
        if current_h > current_w:
            target_h = int(target_w * (current_h / current_h))  # ask to mentor, in the original code it was 
                                                                # target_h = int(self.size * (current_h / current_w))
                                                                # but self.size is a tuple
        elif current_w > current_h:
            target_w = int(target_h * (current_w / current_w))  # same as above

        ppp.input().tensor().set_layout(Layout("NCHW"))

        input_shape = list(meta["input_shape"])

        input_shape[meta["layout"].get_index_by_name("H")] = target_h # ask to mentor, with -1 it was not working
        input_shape[meta["layout"].get_index_by_name("W")] = target_w # ask to mentor, with -1 it was not working

        #ppp.input().tensor().set_shape(input_shape) -> this is not working, ask to mentor
        ppp.input().preprocess().resize(resize_mode_map[self.interpolation], target_h, target_w)
        return [input_shape, Layout("NCHW")]


# CenterCrop transformation
class CenterCrop(Transform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        input_shape = meta["input_shape"]
        source_size = input_shape[2:]
        target_size = self.size

        if target_size[0] > source_size[0] or target_size[1] > source_size[1]:
            raise ValueError("Requested crop size is larger than the input size")
        
        bottom_left = []
        bottom_left.append(int((source_size[0] - target_size[0]) / 2))
        bottom_left.append(int((source_size[1] - target_size[1]) / 2))

        top_right = []
        top_right.append(min(bottom_left[0] + target_size[0], source_size[0] - 1))
        top_right.append(min(bottom_left[1] + target_size[1], source_size[1] - 1))

        bottom_left = [0] * len(input_shape[:-2]) + bottom_left if meta["layout"] == Layout("NCHW") else [0] + bottom_left + [0]
        top_right = list(input_shape[:-2]) + top_right if meta["layout"] == Layout("NCHW") else input_shape[:1] + top_right + input_shape[-1:]
        
        ppp.input().preprocess().crop(bottom_left, top_right)
        # returning the wrond shape? ask to mentor
        return [[input_shape[0], input_shape[1], target_size[0], target_size[1]], meta["layout"]]


# Normalize transformation
class Normalize(Transform):
    """Normalize a float tensor image with mean and standard deviation.
    This transformation does not support PIL Image.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        (TBA: inplace(bool,optional): Bool to make this operation inplace.)

    Returns:
        Tensor: Normalized Tensor image.
    """
    def __init__(self, mean: List, std: List):
        self.mean = mean
        self.std = std

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        ppp.input().preprocess().mean(self.mean).scale(self.std)
        return [meta["input_shape"], meta["layout"]]



# Color conversion transformation
#TODO: to be tested -> how(?)
class ConvertColor(Transform):
    def __init__(self, color_format):
        self.color_format = color_format

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        ppp.input().preprocess().convert_color(self.color_format)
        return [meta["input_shape"], meta["layout"]]


# ToTensor transformation
# need to implement support for Pil image
class ToTensor(Transform):
    """
    convert numpy.ndarray -> Tensor
    TBA support PIL Image
    """
    def __init__(self):
        pass
    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        input_shape = meta["input_shape"]
        layout = meta["layout"]

        ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.RGB)

        if layout == Layout("NHWC"):
            input_shape = f._NHWC_to_NCHW(input_shape)
            layout = Layout("NCHW")
            ppp.input().preprocess().convert_layout(layout)
        ppp.input().preprocess().convert_element_type(Type.f32)
        ppp.input().preprocess().scale(255.0)
        
        return [input_shape, layout]


# TODO: to be tested -> how(?)
class ConvertImageDtype(Transform):
    """
    Convert a tensor image to the given `dtype` and scale the values accordingly
    This function does not support PIL Image.

    Args:
        image (Tensor): Image to be converted
        dtype (dtype): Desired data type of the output

    Returns:
        Tensor: Converted image

    """
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        ppp.input().preprocess().convert_element_type(TORCHTYPE_TO_OVTYPE[self.dtype])
        return [meta["input_shape"], meta["layout"]]


# Pad transformation
class Pad(Transform):
    def __init__(self, padding, fill=0, mode = "constant"):
        self.padding = padding
        self.fill = fill
        self.mode = mode

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        image_dimensions = list(meta["input_shape"][2:])
        layout = meta["layout"]
        torch_padding = self.padding
        pad_mode = self.mode

        if pad_mode == "constant":
            if (isinstance(self.fill, tuple)):
                raise ValueError("Different fill values for R, G, B channels are not supported.")

        pads_begin = [0 for _ in meta["input_shape"]]
        pads_end = [0 for _ in meta["input_shape"]]

       # padding equal on all sides
        if isinstance(torch_padding, int):
            image_dimensions[0] += 2 * torch_padding
            image_dimensions[1] += 2 * torch_padding

            pads_begin[layout.get_index_by_name("H")] = torch_padding
            pads_begin[layout.get_index_by_name("W")] = torch_padding
            pads_end[layout.get_index_by_name("H")] = torch_padding
            pads_end[layout.get_index_by_name("W")] = torch_padding

        # padding different in horizontal and vertical axis
        elif len(torch_padding) == 2:
            image_dimensions[0] += sum(torch_padding)
            image_dimensions[1] += sum(torch_padding)

            pads_begin[layout.get_index_by_name("H")] = torch_padding[1]
            pads_begin[layout.get_index_by_name("W")] = torch_padding[0]
            pads_end[layout.get_index_by_name("H")] = torch_padding[1]
            pads_end[layout.get_index_by_name("W")] = torch_padding[0]

        # padding different on top, bottom, left and right of image
        else:
            image_dimensions[0] += torch_padding[1] + torch_padding[3]
            image_dimensions[1] += torch_padding[0] + torch_padding[2]

            pads_begin[layout.get_index_by_name("H")] = torch_padding[1]
            pads_begin[layout.get_index_by_name("W")] = torch_padding[0]
            pads_end[layout.get_index_by_name("H")] = torch_padding[3]
            pads_end[layout.get_index_by_name("W")] = torch_padding[2]

        @custom_preprocess_function
        def pad_node(output: ov.runtime.Output) -> Callable:
            return ops.pad(
                output,
                pad_mode=pad_mode,
                pads_begin=pads_begin,
                pads_end=pads_end,
                arg_pad_value=np.array(self.fill, dtype=np.uint8) if pad_mode == "constant" else None,
            )
        ppp.input().preprocess().custom(pad_node)
        return [[meta["input_shape"][layout.get_index_by_name("N")], 
                 meta["input_shape"][layout.get_index_by_name("C")],
                 image_dimensions[0], image_dimensions[1]], 
                 meta["layout"]]


# trying custom preprocessing
class abs(Transform):
    def __init__(self):
        pass
    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        ppp.input().preprocess().custom(f.custom_preprocess_abs)
        return [meta["input_shape"], meta["layout"]]