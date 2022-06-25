# Overview
This package provides a MLServer runtime compatible with Openvino. This package has couple features:
1. If server detect that model file is onnx format script will auto convert to openvino format (xml, bin) with dynamic batch size for openvino.
2. Openvino dynamic batch size
3. Grpc Ready
4. V2 Inference Protocol
5. Models metrics

### Why MLserver?
For serving Openvino I choose MLServer because this framework has V2 Inference Protocol (https://kserve.github.io/website/modelserving/inference_api/), grpc and metrics out of the box.

# Install
```sh
pip install mlserver mlserver-openvino
```

# Content Types
If no content type is present on the request or metadata, 
the Openvino runtime will try to decode the payload as a NumPy Array. 
To avoid this, either send a different content type explicitly, or define the correct one as part of your model’s [metadata](https://mlserver.readthedocs.io/en/latest/reference/model-settings.html).

# Models repository
Your models add to models folder.
Accepted files: ["model.xml", "model.onnx"]
```sh
/example
/models/your-model-name/
/tests
setup.py
README.md
```
Training and serve example: https://mlserver.readthedocs.io/en/latest/examples/sklearn/README.html

# Metrics
For download metrics (prometheus) use below links
```sh
GET http://<your-endpoint>/metrics
GET http://0.0.0.0:8080/metrics
```

# Start docker server
```sh
# Build docker image
mlserver build . -t test

# Start server and pass mlserevr_models_dir
docker run -it --rm -e MLSERVER_MODELS_DIR=/opt/mlserver/models/ -p 8080:8080 -p 8081:8081 test
```

# Example queries:
For example script see below files:
```sh
/example/grpc-example.py
/example/rest-example.py
```

# Kserve usage
1. First create one time kserve runtime from file: kserve/cluster-runtime.yaml
2. Create InferenceService from template:
```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "my-openvino-model"
spec:
  predictor:
    model:
      modelFormat:
        name: openvino
      runtime: kserve-mlserver-openvino
      #storageUri: "gs://kfserving-examples/models/xgboost/iris"
      storageUri: https://github.com/myrepo/models/mymodel.joblib?raw=true

```