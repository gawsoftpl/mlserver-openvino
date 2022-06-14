import requests
import json
from mlserver.codecs import  NumpyCodec
from mlserver.types import InferenceResponse
import numpy as np
import time

# Read mnist-onnx-openvino request
with open('./example-mnist-images.json', 'r') as f:
    inference_request_images = json.load(f)

inference_request = {
    "outputs": [{
        "name": "output"
    }],
    "inputs": [{
        "name": "input-0",
        "shape": [20, 28, 28, 1],
        "datatype": "FP32",
        "data": inference_request_images,
    }]
}

# Send to endpoint
endpoint = "http://localhost:8080/v2/models/mnist-onnx-openvino/versions/v0.1.0/infer"
timer = time.time()
response = requests.post(endpoint, json=inference_request)
print(f"Predict time: {time.time() - timer}")

# Parse response from raw
response_payload = InferenceResponse.parse_raw(response.text)

# Convert response
# Mlserver return data with V2 inference protocol standard
# When server return numpy inference response
response_obj = NumpyCodec.decode(response_payload.outputs[0])

for index, result in enumerate(response_obj):
    predicted_index = np.argmax(response_obj[index], axis=0)
    print(f"Letter {index}: {predicted_index}, proba: {round(response_obj[index][predicted_index], 3)} ")
