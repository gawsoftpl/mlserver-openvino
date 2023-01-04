import json
from typing import Any
from mlserver.codecs import Base64Codec, register_input_codec
from mlserver.types import RequestInput
from mlserver.errors import InferenceError
from .lib import unpack, _decompress_base64_gzip_json


@register_input_codec
class JSONGzippedBase64Codec(Base64Codec):
    ContentType = "base64_gzip_json"

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return False

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> Any:  # type: ignore
        try:
            packed = request_input.data.__root__
            unpacked = list(map(_decompress_base64_gzip_json, unpack(packed)))
            return json.loads("[" + ",".join([sentence for sentence in unpacked]) + "]")
        except Exception as e:
            # There are a few things that can go wrong here, e.g. less than 2-D
            # in the array), or input data not compatible with a numpy array
            raise Exception(e)
            raise InferenceError("Invalid input to JSONGzippedBase64") from e

