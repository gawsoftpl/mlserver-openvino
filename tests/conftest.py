from mlserver.settings import ModelSettings, ModelParameters
from mlserver_openvino import OpenvinoRuntime
import os
import pytest
import mlserver.types as types
import joblib

TEST_PATH = os.path.dirname(__file__)

@pytest.fixture
def model_settings() -> ModelSettings:
    return ModelSettings(
        name="ensemble-model",
        parameters=ModelParameters(
            version="v1.2.3",
            uri=os.path.join(TEST_PATH, 'test-data/model-nn.onnx'),
            extra={}
        ),
    )

@pytest.fixture
def model(model_settings: ModelSettings) -> OpenvinoRuntime:
    return OpenvinoRuntime(model_settings)

@pytest.fixture
def inference_request() -> types.InferenceRequest:
    X = joblib.load(os.path.join(TEST_PATH, 'test-data/test_input.joblib'))

    return types.InferenceRequest(
        outputs = [
          types.RequestOutput(
              name="output",
          )
        ],
        inputs=[
            types.RequestInput(
                name="childrens_html_attr_values",
                shape=X['childrens']['attr_values'].shape,
                datatype="INT32",
                data=X['childrens']['attr_values'].flatten(),
                parameters=types.Parameters(content_type="np")
            ),
            types.RequestInput(
                name="childrens_words_inputs",
                shape=X['childrens']['words'].shape,
                datatype="INT32",
                data=X['childrens']['words'].flatten(),
                parameters=types.Parameters(content_type="np")
            ),
            types.RequestInput(
                name="html_attrs_values_input",
                shape=X['elements']['attr_values'].shape,
                datatype="INT32",
                data=X['elements']['attr_values'].flatten(),
                parameters=types.Parameters(content_type="np")
            ),
            types.RequestInput(
                name="parents_html_attr_values",
                shape=X['parents']['attr_values'].shape,
                datatype="INT32",
                data=X['parents']['attr_values'].flatten(),
                parameters=types.Parameters(content_type="np")
            ),
            types.RequestInput(
                name="parents_words_inputs",
                shape=X['parents']['words'].shape,
                datatype="INT32",
                data=X['parents']['words'].flatten(),
                parameters=types.Parameters(content_type="np")
            ),

            types.RequestInput(
                name="words_inputs",
                shape=X['elements']['words'].shape,
                datatype="INT32",
                data=X['elements']['words'].flatten(),
                parameters=types.Parameters(content_type="np")
            ),
        ]
    )