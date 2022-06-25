FROM python:3.8.13-slim-bullseye
USER 1000
ADD requirements.txt .
RUN pip install -r requirements.txt
COPY mlserver_openvino .
ENTRYPOINT python convert.py /mnt/model/model.onnx /mnt/openvino
