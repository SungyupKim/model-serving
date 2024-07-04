import requests
import json
import torch
# Define the endpoint
#endpoint = "http://inferenceservices4.hdx-100-hd-com.inference.terraform-ktc-develop.matilda-mzc.com/v1/models/mnist:predict"
#endpoint = "http://inferenceservices3.hdx-100-hd-com.inference.terraform-ktc-develop.matilda-mzc.com/v1/models/inferenceservices3:predict"
endpoint = "http://inference-torch-predictor-default.hdx-100-hd-com.inference.terraform-ktc-develop.matilda-mzc.com/v1/models/mnist:predict"

sample_input = torch.randn(1, 1, 28, 28)

# Convert the sample input to a list (for JSON serialization)
sample_input_list = sample_input.numpy().tolist()

# Prepare the payload
payload = {
    "instances": sample_input_list
}

# Send the POST request
response = requests.post(endpoint, json=payload)

# Check the response
if response.status_code == 200:
    print("Prediction response:", response.json())
else:
    print("Failed to get prediction. Status code:", response.status_code)
    print("Response:", response.text)
