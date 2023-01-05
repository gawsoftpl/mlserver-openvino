import pytest
import numpy as np
import base64
import gzip
from typing import Any
from mlserver_openvino.codecs.numpy_gzip import NumpyGzipCodec
from mlserver.types import RequestInput, ResponseOutput

def _convert_numpy(data: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(data.tobytes())).decode('ascii')

def _decompress_numpy(data: str, type_name: Any, dsize: Any) -> np.ndarray:
    return np.frombuffer(gzip.decompress(base64.b64decode(data)), type_name).reshape(dsize)

#raise Exception(_decompress_numpy("H4sIAMjBtWMC/2NkgAAmKM0MpQGHjcsrGAAAAA==", np.int64, [1, 3]))
#raise Exception(_convert_numpy(np.array([b"\x01\x02"], dtype=np.dtype(bytes))))

@pytest.mark.parametrize(
    "payload, expected",
    [(np.array([1, 2, 3]), False)],
)
def test_can_encode(payload: Any, expected: bool):
    assert NumpyGzipCodec.can_encode(payload) == expected

@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            np.array([[1], [2], [3]], np.int64),
            ResponseOutput(name="foo", shape=[3, 1], data=["H4sIAKLDtWMC/2NkgAAmKM0MpQGHjcsrGAAAAA=="], datatype="INT64"),
        ),
        (
            np.array([[1, 2], [3, 4]], np.int32),
            ResponseOutput(
                name="foo", shape=[2, 2], data=['H4sIALjFtWMC/2NkYGBgAmJmIGYBYgDv1AWvEAAAAA=='], datatype="INT32"
            ),
        ),
        (
            np.array([["foo"], ["bar"]], dtype=str),
            ResponseOutput(
                name="foo", shape=[2, 1], data=["H4sIABLGtWMC/0tjYGDIh+IkIE4E4iIgBgCbzNg2GAAAAA=="], datatype="BYTES"
            ),
        ),
    ],
)
def test_encode_output(payload: np.ndarray, expected: ResponseOutput):
    response_output = NumpyGzipCodec.encode_output(name="foo", payload=payload)

    response_output_np = _decompress_numpy(response_output.dict()['data'][0], payload.dtype, payload.shape)
    expected_output_np = _decompress_numpy(expected.dict()['data'][0], payload.dtype, expected.shape)

    # Decom
    assert (response_output_np == expected_output_np).all()


@pytest.mark.parametrize(
    "request_input, expected",
    [
        (
            RequestInput(name="foo", shape=[3, 1], data=["H4sIAKLDtWMC/2NkgAAmKM0MpQGHjcsrGAAAAA=="], datatype="INT64"),
            np.array([[1], [2], [3]], np.int64),
        ),
        (
            RequestInput(name="foo", shape=[2, 2], data=["H4sIALjFtWMC/2NkYGBgAmJmIGYBYgDv1AWvEAAAAA=="], datatype="INT32"),
            np.array([[1, 2], [3, 4]], np.int32),
        ),
        # (
        #     RequestInput(name="foo", shape=[2,1], data=["H4sIABLGtWMC/0tjYGDIh+IkIE4E4iIgBgCbzNg2GAAAAA=="], datatype="BYTES"),
        #     np.array([["foo"],["bar"]], dtype=str),
        # ),

    ],
)
def test_decode_input(request_input: RequestInput, expected: np.ndarray):
    decoded = NumpyGzipCodec.decode_input(request_input)
    np.testing.assert_array_equal(decoded, expected)


@pytest.mark.parametrize(
    "request_input",
    [
        RequestInput(name="foo", shape=[3], data=["H4sIAAnNtWMC/2NkYGBgAmJmIAYAkyLgsAwAAAA="], datatype="INT32"),
        RequestInput(name="foo", shape=[2, 2], data=["H4sIACnNtWMC/2NkYGBgAmJmIGYBYgDv1AWvEAAAAA=="], datatype="INT32"),
        RequestInput(name="foo", shape=[2], data=["H4sIANPNtWMC/2NgaLBnYGBwAAB2pT8uCAAAAA=="], datatype="FP32"),
        #RequestInput(name="foo", shape=[2, 1], data=["H4sIAArOtWMC/2NkAgCSQsy2AgAAAA=="], datatype="BYTES"),
       # RequestInput(name="foo", shape=[2], data=["foo", "bar"], datatype="BYTES"),
    ],
)
def test_encode_input(request_input):
    decoded = NumpyGzipCodec.decode_input(request_input)
    response_output = NumpyGzipCodec.encode_output(name="foo", payload=decoded)

    request_input_result = NumpyGzipCodec.encode_input(name="foo", payload=decoded)
    assert response_output.datatype == request_input_result.datatype
    assert response_output.shape == request_input_result.shape
    assert response_output.data == request_input_result.data
    assert request_input_result.parameters.content_type == NumpyGzipCodec.ContentType


@pytest.mark.parametrize(
    "payload, expected",
    [
        (np.array([1, 2, 3]), np.array([[1], [2], [3]])),
        (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
        (np.array([1.0, 2.0]), np.array([[1.0], [2.0]])),
        # (
        #     np.array([[b"\x01"], [b"\x02"]], dtype=bytes),
        #     np.array([[b"\x01"], [b"\x02"]], dtype=bytes),
        # ),
    ],
)
def test_decode_output(payload, expected):
    response_output = NumpyGzipCodec.encode_output(name="foo", payload=payload)
    output_response_decoded = NumpyGzipCodec.decode_output(response_output)
    np.testing.assert_array_equal(output_response_decoded, expected)

