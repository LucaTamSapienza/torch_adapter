from src.torch_adapter.transforms import *
from openvino.preprocess import  ColorFormat
import numpy as np
#import openvino as ov

# Defining the preprocessing tranformations
preprocess = Compose([
    # Resize(256, 256),
    # ToTensor()
    # ConvertColor(ColorFormat.RGB),
    Scale(2.0),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Creating preprocessing model
data = np.ones((1, 3, 224, 224), dtype=np.float32)
print("data = ", data)

# preprocessing...
preprocess_model = preprocess([data.shape], [data.dtype])
print("preprocess_model = ", preprocess_model)
result = preprocess_model(data)[0]
print("result = ", result)