import requests
import json
import numpy as np
import tensorflow as tf
import ssl
# Replace with your actual test image (normalize it as well)
ssl._create_default_https_context = ssl._create_unverified_context
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

test_image = x_test[0].reshape(1, 28, 28).tolist()  # Convert a test image to list for JSON serialization

# TensorFlow Serving URL
url = 'https://proxy1.aiserv.ktcloud.com:10396/v1/models/default:predict'
headers = {"content-type": "application/json"}

# Prepare the data in the format required by TensorFlow Serving
data = json.dumps({"signature_name": "serving_default", "instances": test_image})

# Send the POST request to TensorFlow Serving
response = requests.post(url, data=data, headers=headers, verify=False)
predictions = json.loads(response.text)

# Extract and print the predicted class
predicted_class = np.argmax(predictions['predictions'][0])
print(f"Predicted class: {predicted_class}")
