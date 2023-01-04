from typing import Any, List, Union
from mlserver.codecs import InputCodec, register_input_codec, register_request_codec
from mlserver.codecs.numpy import to_dtype, to_datatype
from mlserver.codecs.lists import ListElement
from mlserver.codecs.utils import inject_batch_dimension, SingleInputRequestCodec
from mlserver.types import RequestInput, ResponseOutput, Parameters
from mlserver.errors import InferenceError
from .lib import unpack
import base64
import gzip
import numpy as np
import binascii

_Base64StrCodec = "ascii"

def _ensure_bytes(elem: ListElement) -> bytes:
    if isinstance(elem, str):
        return elem.encode(_Base64StrCodec)
    return elem

def _encode_base64(elem: ListElement, use_bytes: bool) -> Union[bytes, str]:
    as_bytes = _ensure_bytes(elem)
    b64_encoded = base64.b64encode(as_bytes)
    if use_bytes:
        return b64_encoded

    return b64_encoded.decode(_Base64StrCodec)


def _decode_base64(elem: ListElement) -> bytes:
    as_bytes = _ensure_bytes(elem)

    # Check that the input is valid base64.
    # Otherwise, convert into base64.
    try:
        return base64.b64decode(as_bytes, validate=True)
    except binascii.Error:
        return as_bytes

def _encode_data(data: np.ndarray) -> list:
    return [_encode_base64(gzip.compress(data.tobytes()), False)]

def _decompress_numpy(body: str) -> bytes:
    return gzip.decompress(base64.b64decode(body))

@register_input_codec
class NumpyGzipCodec(InputCodec):
    ContentType = "np_gzip"

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return False

    @classmethod
    def decode_output(cls, response_output: ResponseOutput) -> np.ndarray:
        return cls.decode_input(response_output)  # type: ignore

    @classmethod
    def encode_output(cls, name: str, payload: np.ndarray, **kwargs) -> ResponseOutput:

        datatype = to_datatype(payload.dtype)
        shape = inject_batch_dimension(list(payload.shape))

        return ResponseOutput(
            name=name,
            datatype=datatype,
            shape=shape,
            data=_encode_data(payload),
        )

    @classmethod
    def encode_input(cls, name: str, payload: np.ndarray, **kwargs) -> RequestInput:
        output = cls.encode_output(name=name, payload=payload)

        return RequestInput(
            name=output.name,
            datatype=output.datatype,
            shape=output.shape,
            data=output.data,
            parameters=Parameters(content_type=cls.ContentType),
        )


    @classmethod
    def decode_input(cls, request_input: RequestInput) -> Any:  # type: ignore
        try:
            dtype = to_dtype(request_input)

            packed = request_input.data.__root__
            unpacked = map(_decompress_numpy, unpack(packed))
            decoded = [np.frombuffer(sentence, dtype) for sentence in unpacked]
            return np.concatenate(decoded).reshape(request_input.shape)

        except Exception as e:
            # There are a few things that can go wrong here, e.g. less than 2-D
            # in the array), or input data not compatible with a numpy array
            raise InferenceError(e) from e

@register_request_codec
class NumpyRequestCodec(SingleInputRequestCodec):
    InputCodec = NumpyGzipCodec
    ContentType = NumpyGzipCodec.ContentType