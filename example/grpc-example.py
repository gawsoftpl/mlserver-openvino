import numpy as np
import time
import grpc
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import json
import mlserver.grpc.converters as converters
import mlserver.types as types

# Read mnist-onnx-openvino request
with open('./example-mnist-images.json', 'rb') as f:
    inference_request_images = json.load(f)

# Prepare grpc request
inference_request_g = converters.ModelInferRequestConverter.from_types(
    types.InferenceRequest(
        outputs= [
               types.RequestOutput(
                   name="output",
                   parameters=types.Parameters(content_type="np")
               )
        ],
        inputs=[
            types.RequestInput(
                name="input-0",
                shape=[20, 28, 28, 1],
                datatype="FP32",
                data=inference_request_images,
                parameters=types.Parameters(content_type="np")
            )
        ]
    ),
        model_name='mnist-onnx-openvino',
        model_version='v0.1.0'
    )

# Connect with grpc channel
grpc_channel = grpc.insecure_channel("127.0.0.1:8081")
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

# Send inference method request
timer = time.time()
response_payload = grpc_stub.ModelInfer(inference_request_g)

print(f"Predict time: {time.time() - timer}")

response_np = np.array(response_payload.outputs[0].contents.fp32_contents).reshape(response_payload.outputs[0].shape)

for index, result in enumerate(response_np):
    predicted_index = np.argmax(response_np[index], axis=0)
    print(f"Letter {index}: {predicted_index}, proba: {round(response_np[index][predicted_index], 3)} ")
