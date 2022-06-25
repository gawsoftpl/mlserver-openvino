build-docker-convert:
	docker image build -t convert-onnx-to-openvino -f converter_docker/Dockerfile .
