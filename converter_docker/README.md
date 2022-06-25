# Overview
This docker image convert onnx file to openvino format

## Build image
```sh
make build-docker-convert
```

# Convert 
```sh
docker run -it --rm -v `pwd`/models/mnist-onnx-openvino/model.onnx:/mnt/model/model.onnx -v /tmp/output:/mnt/openvino gawsoft/mlserver-openvino-onnx-converter
```