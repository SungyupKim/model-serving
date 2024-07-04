import torch
import torch.nn as nn
from ts.torch_handler.base_handler import BaseHandler

class NoPrePostProcessingHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
        print(context.system_properties)
        #model_dir = "/mnt/models/model-store" #context.system_properties.get("model_store")
        #print("dir : " + str(model_dir))
        self.model = self.load_model(context.system_properties.get("model_dir"))
        self.model.eval()
        #self.model = model

    def load_model(self, model_dir):
        #model = torch.load(model_dir+"/mnist.pt")
        model = torch.jit.load(model_dir+"/mnist.pt")
        return model
    #    import os
    #    from ts.torch_handler.base_handler import _load_pickled_model

    #    model_path = os.path.join(model_dir, "model.pth")
    #    model = _load_pickled_model(model_path)
    #    return model

    def preprocess(self, data):
        # No preprocessing
        return data

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            print(data)
            inputs = torch.tensor(data[0]["body"]["instances"][0], device=self.device)
            outputs = self.model(inputs)
            print(outputs)
        import json
        return json.dumps(outputs.tolist())

    def postprocess(self, data):
        # No postprocessing
        return data
