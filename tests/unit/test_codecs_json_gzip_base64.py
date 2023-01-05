import pytest
import numpy as np
from typing import Any, List
from mlserver_openvino.codecs.json_gzip_base64 import JSONGzippedBase64Codec, _decompress_base64_gzip_json, _compress_base64_gzip_json
from mlserver.types import RequestInput, ResponseOutput

@pytest.mark.parametrize(
    "request_input, expected",
    [
        (
            RequestInput(name="foo", shape=[1, 1], data=["H4sIAGzVtWMC/6tWSlSyUogGkjoKSkkgIlkpthYAn3kbCRYAAAA="], datatype="BYTES"),
            [{
               "a": ["a","b","c"]
            }]
        ),
        (
            RequestInput(name="foo", shape=[1, 1], data=["H4sIAIXVtWMC/4tWSlTSUVBKAhHJQCLaUEfBSEfBODYWAENlVGEaAAAA"], datatype="BYTES"),
            [["a","b","c",[1,2,3]]],
        ),

    ],
)
def test_decode_input(request_input: RequestInput, expected: np.ndarray):
    decoded = JSONGzippedBase64Codec.decode_input(request_input)
    assert decoded == expected

@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            [{"a":1,"b":[1,2],"b":3}],
            ResponseOutput(name="foo", shape=[1, 1], data=["H4sIAM3WtWMC/6tWSlSyUjDUUVBKAtLGtQBeq64yEAAAAA=="], datatype="BYTES"),
        ),
        (
            [[1,2,3,4]],
            ResponseOutput(
                name="foo", shape=[1, 1], data=['H4sIAOHWtWMC/4s21FEw0lEw1lEwiQUAHIXJWgwAAAA='], datatype="BYTES"
            ),
        ),
(
            [[1,2,3,4], [1,2,3,8]],
            ResponseOutput(
                name="foo", shape=[2, 1], data=['H4sIAOHWtWMC/4s21FEw0lEw1lEwiQUAHIXJWgwAAAA=', 'H4sIAMbXtWMC/4s21FEw0lEw1lGwiAUAEMp89gwAAAA='], datatype="BYTES"
            ),
        ),
    ],
)
def test_encode_output(payload: List[Any], expected: ResponseOutput):
    response_output = JSONGzippedBase64Codec.encode_output(name="foo", payload=payload)
    response_output_np = _decompress_base64_gzip_json(response_output.dict()['data'][0])
    expected_output_np = _decompress_base64_gzip_json(expected.dict()['data'][0])

    # Decom
    assert response_output_np  == expected_output_np

@pytest.mark.parametrize(
    "request_input",
    [
        RequestInput(name="foo", shape=[1,1], data=["H4sIAM3WtWMC/6tWSlSyUjDUUVBKAtLGtQBeq64yEAAAAA=="], datatype="BYTES"),
        RequestInput(name="foo", shape=[2, 1], data=["H4sIAOHWtWMC/4s21FEw0lEw1lEwiQUAHIXJWgwAAAA=","H4sIAMbXtWMC/4s21FEw0lEw1lGwiAUAEMp89gwAAAA="], datatype="BYTES"),

    ],
)
def test_encode_input(request_input):
    decoded = JSONGzippedBase64Codec.decode_input(request_input)
    response_output = JSONGzippedBase64Codec.encode_output(name="foo", payload=decoded)

    request_input_result = JSONGzippedBase64Codec.encode_input(name="foo", payload=decoded)
    assert response_output.datatype == request_input_result.datatype
    assert response_output.shape == request_input_result.shape
    assert response_output.data == request_input_result.data
    assert request_input_result.parameters.content_type == JSONGzippedBase64Codec.ContentType

@pytest.mark.parametrize(
    "payload, expected",
    [
        ([1, 2, 3], [[1], [2], [3]]),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ([1.0, 2.0], [[1.0], [2.0]]),
    ],
)
def test_decode_output(payload, expected):
    response_output = JSONGzippedBase64Codec.encode_output(name="foo", payload=payload)
    output_response_decoded = JSONGzippedBase64Codec.decode_output(response_output)
    output_response_decoded == expected