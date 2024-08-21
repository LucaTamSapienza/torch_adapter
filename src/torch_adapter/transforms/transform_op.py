from typing import Dict, List, Tuple, Union, Callable, Sequence

import numpy as np
import openvino as ov
import openvino.runtime.opset14 as ops
import torch
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Layout, Type
from openvino.runtime.utils.decorators import custom_preprocess_function
from torchvision.transforms import InterpolationMode

from . import util as f


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
    def __init__(self) -> None:
        pass
    def __call__(self, ppp: PrePostProcessor) -> List:
        pass


# Resize transformation
class Resize(Transform):
    """Resize the input image to the given size.

    Args:
        size (Tuple or int): Desired output size. If size is a Tuple like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.

        max_size (int, optional): Not Supported

    """
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation=InterpolationMode.BILINEAR, max_size = 0):
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

        target_h, target_w = f._setup_size(self.size, "Incorrect size type for Resize operation")

        # rescale the smaller image edge
        current_h, current_w = meta["input_shape"][2:4] if meta["layout"] == Layout("NCHW") else meta["input_shape"][1:3]

        if isinstance(self.size, int): # is this check needed?
            if current_h > current_w:
                target_h = int(self.size * (current_h / current_w))
            elif current_w > current_h:
                target_w = int(self.size * (current_w / current_h))

        ppp.input().tensor().set_layout(Layout("NCHW"))

        input_shape = list(meta["input_shape"])

        input_shape[meta["layout"].get_index_by_name("H")] = target_h
        input_shape[meta["layout"].get_index_by_name("W")] = target_w

        #ppp.input().tensor().set_shape(input_shape)
        ppp.input().preprocess().resize(resize_mode_map[self.interpolation], target_h, target_w)
        return [input_shape, Layout("NCHW")]


# CenterCrop transformation
class CenterCrop(Transform):
    """Crops the given image at the center.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (Tuple or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        input_shape = meta["input_shape"]
        source_size = input_shape[2:]
        target_size = f._setup_size(self.size, "Incorrect size type for CenterCrop operation")

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
        
        return [[input_shape[0], input_shape[1], target_size[0], target_size[1]], meta["layout"]]


# Normalize transformation
class Normalize(Transform):
    """Normalize a float tensor image with mean and standard deviation.
    This transformation does not support PIL Image.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (N, C, H, W) to be normalized.
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


# ToTensor transformation
class ToTensor(Transform):
    """Used for PIL Image only. Scale the value of the Image to [0 ,1]

    Args:
        None
    
    Returns:
        Tensor: Scaled tensor image.
    """
    def __init__(self):
        pass
    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:

        # ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.RGB)

        ppp.input().preprocess().convert_element_type(Type.f32)
        ppp.input().preprocess().scale(255.0)
        
        return [meta["input_shape"], meta["layout"]]


# ConvertImageDtype transformation
class ConvertImageDtype(Transform):
    """Convert a tensor image to the given ``dtype`` and scale the values accordingly.

    Args:
        dtype (torch.dtype): Desired data type of the output
        .. note::
            torch.dtype are mapped to the corresponding ``dtype`` of openvino

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        ppp.input().preprocess().convert_element_type(TORCHTYPE_TO_OVTYPE[self.dtype])
        return [meta["input_shape"], meta["layout"]]


# Pad transformation
class Pad(Transform):
    """Pad the given image on all sides with the given "pad" value.

        Args:
            padding (int or sequence): Padding on each border. If a single int is provided this
                is used to pad all borders. If sequence of length 2 is provided this is the padding
                on left/right and top/bottom respectively. If a sequence of length 4 is provided
                this is the padding for the left, top, right and bottom borders respectively.

            fill (number): Pixel fill value for constant fill. Default is 0.

            padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
                Default is constant.

                - constant: pads with a constant value, this value is specified with fill

                - edge: pads with the last value at the edge of the image.
                If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

                - reflect: pads with reflection of image without repeating the last value on the edge.
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

                - symmetric: pads with reflection of image repeating the last value on the edge.
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding: Sequence, fill=0, padding_mode = "constant"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, ppp: PrePostProcessor, meta: Dict) -> List:
        image_dimensions = list(meta["input_shape"][2:])
        layout = meta["layout"]
        torch_padding = self.padding
        pad_mode = self.padding_mode

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