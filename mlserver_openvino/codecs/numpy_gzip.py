from typing import Any
from mlserver.codecs import Base64Codec, register_input_codec
from mlserver.codecs.numpy import to_dtype
from mlserver.types import RequestInput
from mlserver.errors import InferenceError
from .lib import unpack
import base64
import gzip
import numpy as np

def decompress_numpy(body):
    return gzip.decompress(base64.b64decode(body))

@register_input_codec
class NumpyGzipCodec(Base64Codec):
    ContentType = "np_gzip"

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return True

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> Any:  # type: ignore
        try:
            dtype = to_dtype(request_input)
            packed = request_input.data.__root__
            unpacked = map(decompress_numpy, unpack(packed))
            decoded = [np.frombuffer(sentence, dtype) for sentence in unpacked]
            return np.concatenate(decoded).reshape(request_input.shape)

        except Exception as e:
            # There are a few things that can go wrong here, e.g. less than 2-D
            # in the array), or input data not compatible with a numpy array
            raise Exception(e)
            raise InferenceError("Invalid input to JSONGzippedBase64") from e

