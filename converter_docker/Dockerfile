FROM python:3.8.13-slim-bullseye

ADD requirements.txt .
RUN pip install -r requirements.txt

USER 1000

COPY mlserver_openvino mlserver_openvino
COPY mlserver_openvino/convert.py convert.py

ENTRYPOINT python convert.py /mnt/model/model.onnx /mnt/openvino
