import openvino as ov

class AdapterModel():
    # model = AdapterModel(torch.hub.load(...))
    def __init__(self, torch_model):
        self.torch_model = torch_model
        self.ov_model = None
        self.final_model = None

    # model.eval()
    # will do two things:
    # - convert the model to an openvino model 
    # - compile the model for the inference
    def eval(self):
        self.ov_model = ov.convert_model(self.torch_model)
        self.final_model = ov.Core().compile_model(self.ov_model, "CPU")
    
    # model(input_tensor)
    # will do inference on the model
    def __call__(self, input_tensor):
        return self.final_model(input_tensor, share_inputs=True, share_outputs=True)
