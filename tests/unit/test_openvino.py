from mlserver_openvino import OpenvinoRuntime
import os
import pytest
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.errors import InferenceError

test_path = os.path.dirname(__file__)

def test_convert_onnx_to_openvino():
    OpenvinoRuntime.convert_onnx(os.path.join(test_path, '../test-data/model-nn.onnx'))
    assert os.path.exists('/tmp/openvino/onnx_model.xml')
    assert os.path.exists('/tmp/openvino/onnx_model.bin')

@pytest.mark.asyncio
async def test_load_model(model: OpenvinoRuntime):
    assert await model.load()

@pytest.mark.asyncio
async def test_predict_wrong_input(model: OpenvinoRuntime, inference_request: InferenceRequest):
    await model.load()
    inference_request.inputs[0].name="abc"
    with pytest.raises(InferenceError):
        await model.predict(inference_request)

@pytest.mark.asyncio
async def test_predict_wrong_output(model: OpenvinoRuntime, inference_request: InferenceRequest):
    await model.load()
    inference_request.outputs[0].name="abc"
    with pytest.raises(InferenceError):
        await model.predict(inference_request)


@pytest.mark.asyncio
async def test_predict(model: OpenvinoRuntime, inference_request: InferenceRequest):
    await model.load()
    response = await model.predict(inference_request)
    assert isinstance(response, InferenceResponse)
    assert len(response.outputs) == 1
    assert response.outputs[0].shape == [2835, 4]