import numpy as np
from openvino.preprocess import PrePostProcessor
import openvino as ov
import openvino.runtime.opset14 as ops
from openvino.runtime import Layout
from src.torch_adapter.transforms.util import create_empty_model


# Base class for all Composition classes
class Composition:
    def __init__(self):
        self._compiled_model = None
        self._last_data = None

    def _needs_recompile(self, data):
        if self._last_data is None:
            self._last_data = data
            return True
        elif self._last_data.shape != data.shape or self._last_data.dtype != data.dtype:
            self._last_data = data
            return True
        return False

    def _compile_model(self, transforms):
        model = create_empty_model([self._last_data.shape], [self._last_data.dtype])
        ppp = PrePostProcessor(model)

        # Set the layout of the input tensor to NCHW by default
        ppp.input().tensor().set_layout(Layout("NCHW"))

        for transform in transforms:
            transform(ppp)

        final_model = ppp.build()
        self._compiled_model = ov.Core().compile_model(final_model, "CPU")


# Compose class is used to apply a list of transforms sequentially
class Compose(Composition):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, data):
        if self._needs_recompile(data) or self._compiled_model is None:
            self._compile_model(self.transforms)

        return self._compiled_model(data, share_inputs=True, share_outputs=True)


# RandomApply class is used to apply a list of transforms randomly with a given probability p (0.5 by default)
class RandomApply(Composition):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p
    
    # override the _compile_model inherited from Composition because the implementation is different
    def _compile_model(self):
        model = create_empty_model([self._last_data.shape], [self._last_data.dtype])
        ppp = PrePostProcessor(model)

        ppp.input().tensor().set_layout(Layout("NCHW"))
        if np.random.random() < self.p:
            for transform in self.transforms:
                transform(ppp)
        final_model = ppp.build()
        self._compiled_model = ov.Core().compile_model(final_model, "CPU")

    def __call__(self, data):
        if self._needs_recompile(data) or self._compiled_model is None:
            self._compile_model()

        return self._compiled_model(data, share_inputs=True, share_outputs=True)


# RandomOrder class is used to apply a list of transforms in a random order
class RandomOrder(Composition):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    # Same logic as RandomApply
    def _compile_model(self):
        model = create_empty_model([self._last_data.shape], [self._last_data.dtype])
        ppp = PrePostProcessor(model)

        ppp.input().tensor().set_layout(Layout("NCHW"))
        order = np.random.permutation(len(self.transforms))

        for i in order:
            self.transforms[i](ppp)
        final_model = ppp.build()
        self._compiled_model = ov.Core().compile_model(final_model, "CPU")

    def __call__(self, data):
        if self._needs_recompile(data) or self._compiled_model is None:
            self._compile_model()

        return self._compiled_model(data, share_inputs=True, share_outputs=True)
