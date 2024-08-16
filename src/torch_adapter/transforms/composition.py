import numpy as np
from openvino.preprocess import PrePostProcessor
import openvino as ov
from openvino.runtime import Layout
from .util import create_empty_model, _NHWC_to_NCHW
from PIL import Image


# Base class for all Composition classes
class Composition:
    def __init__(self):
        self._compiled_model = None
        self._last_data = None

    def _needs_recompile(self, data) -> bool:
        if self._last_data is None:
            self._last_data = data
            return True
        elif self._last_data.shape != data.shape or self._last_data.dtype != data.dtype:
            self._last_data = data
            return True
        return False

    def _compile_model(self, transforms):
        if isinstance(self._last_data, Image.Image):
            self._last_data = np.array(self._last_data, dtype=np.float32)
            self._last_data = np.expand_dims(self._last_data, axis=0)
            self._last_data.dtype = np.float32
            self._last_data.shape = _NHWC_to_NCHW(list(self._last_data.shape))
            # print(self._last_data.shape)
        
        model = create_empty_model([self._last_data.shape], [self._last_data.dtype])
        ppp = PrePostProcessor(model)

        # Set the layout of the input tensor to NCHW by default
        ppp.input().tensor().set_layout(Layout("NCHW"))

        # last_shape = self._last_data.shape
        meta = {"input_shape": self._last_data.shape, "layout": Layout("NCHW")}
        for transform in transforms:
            temp = transform(ppp, meta)
            meta["input_shape"] = temp[0]
            meta["layout"] = temp[1]

        # Build the preprocessing pipeline
        final_model = ppp.build()
        self._compiled_model = ov.Core().compile_model(final_model, "CPU")
        # ov.serialize(final_model, "temp.xml") # save the model (to open it with netron for example)


# Compose class is used to apply a list of transforms sequentially
class Compose(Composition):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     transforms.CenterCrop(224),
        >>>     transforms.pad((1, 1)),
        >>> ])

    Note::
        The Compose Class create an empty model (with desired shape and data types) 
        and stack all the transformation on top of it.

    """
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
    # @profile -> uncomment this to profile the function
    def __call__(self, data):
        if self._needs_recompile(data) or self._compiled_model is None:
            #print("Recompiling...")
            self._compile_model(self.transforms)

        return self._compiled_model(self._last_data, share_inputs=True, share_outputs=True)


# RandomApply class is used to apply a list of transforms randomly with a given probability p (0.5 by default)
class RandomApply(Composition):
    """Apply a list of transformations randomly with a given probability p.
    
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
        p (float): probability that the transformation is applied.
        
    Example:
        >>> transforms.RandomApply([
        >>>     transforms.Resize((256, 270)),
        >>>     transforms.CenterCrop((224, 224)),
        >>>     transforms.pad((1, 1)),
        >>> ], p=0.5)
    
    In this example, each of the transformations is applied with a probability of 0.5.
    """
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p
    
    # override the _compile_model inherited from Composition
    def _compile_model(self, transforms):
        model = create_empty_model([self._last_data.shape], [self._last_data.dtype])
        ppp = PrePostProcessor(model)

        ppp.input().tensor().set_layout(Layout("NCHW"))

        meta = {"input_shape": self._last_data.shape, "layout": Layout("NCHW")}

        if np.random.random() < self.p:
            for transform in transforms:
                temp = transform(ppp, meta)
                meta["input_shape"] = temp[0]
                meta["layout"] = temp[1]

        final_model = ppp.build()
        self._compiled_model = ov.Core().compile_model(final_model, "CPU")

    def __call__(self, data):
        if self._needs_recompile(data) or self._compiled_model is None:
            self._compile_model(self.transforms)

        return self._compiled_model(data, share_inputs=True, share_outputs=True)


# RandomOrder class is used to apply a list of transforms in a random order
class RandomOrder(Composition):
    """
    .. note::
        TBA: description
    """
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    # Same logic as RandomApply
    def _compile_model(self, transforms):
        model = create_empty_model([self._last_data.shape], [self._last_data.dtype])
        ppp = PrePostProcessor(model)

        ppp.input().tensor().set_layout(Layout("NCHW"))
        order = np.random.permutation(len(self.transforms))

        meta = {"input_shape": self._last_data.shape, "layout": Layout("NCHW")}

        for i in order:
            temp = transforms[i](ppp, meta)
            meta["input_shape"] = temp[0]
            meta["layout"] = temp[1]
        final_model = ppp.build()
        self._compiled_model = ov.Core().compile_model(final_model, "CPU")

    def __call__(self, data):
        if self._needs_recompile(data) or self._compiled_model is None:
            self._compile_model(self.transforms)

        return self._compiled_model(data, share_inputs=True, share_outputs=True)
