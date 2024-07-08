import torch
import torch.nn as nn
from ts.torch_handler.base_handler import BaseHandler

class NoPrePostProcessingHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
        self.model = self.load_model(context.system_properties.get("model_dir"))
        self.model.eval()
        self.initialized = True

    def load_model(self, model_dir):
        model = torch.jit.load(model_dir+"/mnist.pt")
        return model

    def preprocess(self, data):
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
        results = []
        for output in data:
            _, predicted = torch.max(output, 0)
            results.append(predicted.item())
        return results
