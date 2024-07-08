import torch
import torch.nn as nn
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image

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
        #images = []
        #for row in data:
        #    image = row.get("data") or row.get("body")
        #    if isinstance(image, (bytearray, bytes)):
        #        image = Image.open(io.BytesIO(image)).convert('L')
        #        image = self.transform(image)
        #        images.append(image)
        #images = torch.stack(images).to(self.device)
        return data

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            #outputs = self.model(data)
            #print(data)
	    #for row in data:
            #    image = row.get("data") or row.get("body")
            #    input =  torch.tensor(data[0]["body"]["instances"][0], device=self.device)
            inputs = torch.tensor(data[0]["body"]["instances"], device=self.device)
            outputs = self.model(inputs)
            #print(outputs)
            outputs = outputs.tolist()

        return [{"result" : [outputs]}]

    def postprocess(self, data):
        # No postprocessing
        #results = []
        #for output in data:
        #_, predicted = torch.max(data)
        #results.append(predicted.item())
        return data
