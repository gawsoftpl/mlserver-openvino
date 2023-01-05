import json
from typing import Any, List
from mlserver.codecs import StringCodec, Base64Codec, register_input_codec, register_request_codec
from mlserver.codecs.utils import SingleInputRequestCodec
from mlserver.types import RequestInput, Parameters, ResponseOutput
from mlserver.errors import InferenceError
from .lib import unpack
import base64
import gzip

def _decompress_base64_gzip_json(body: str) -> str:
    return gzip.decompress(base64.b64decode(body)).decode('utf-8')

def _compress_base64_gzip_json(data: Any) -> str:
    return base64.b64encode(gzip.compress(json.dumps(data).encode('utf-8'))).decode('utf-8')

@register_input_codec
class JSONGzippedBase64Codec(Base64Codec):
    ContentType = "base64_gzip_json"

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return False

    @classmethod
    def encode_output(cls, name: str, payload: List[Any], **kwargs) -> ResponseOutput:
        shape = [len(payload), 1]
        print(list(map(_compress_base64_gzip_json, payload)))
        return ResponseOutput(
            name=name,
            datatype="BYTES",
            shape=shape,
            data=list(map(_compress_base64_gzip_json, payload)),
            parameters=Parameters(content_type=cls.ContentType),
        )

    @classmethod
    def encode_input(cls, name: str, payload: List[Any], **kwargs) -> RequestInput:
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
            packed = request_input.data.__root__
            unpacked = list(map(_decompress_base64_gzip_json, unpack(packed)))
            print(unpacked)
            return json.loads("[" + ",".join([sentence for sentence in unpacked]) + "]")
        except Exception as e:
            # There are a few things that can go wrong here, e.g. less than 2-D
            # in the array), or input data not compatible with a numpy array
            raise InferenceError("Invalid input to JSONGzippedBase64") from e

@register_request_codec
class JSONGzippedBase64RequestCodec(SingleInputRequestCodec):
    InputCodec = JSONGzippedBase64Codec
    ContentType = JSONGzippedBase64Codec.ContentType

@register_input_codec
class JSONCodec(StringCodec):
    ContentType = "json"

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return False

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> Any:  # type: ignore
        try:
            unpacked = super().decode_input(request_input)
            # out = []
            # for k in unpacked:
            #     out.append(json.loads(k))
            # return out
            return json.loads("[" + ",".join([sentence for sentence in unpacked]) + "]")

        except Exception as e:
            # There are a few things that can go wrong here, e.g. less than 2-D
            # in the array), or input data not compatible with a numpy array
            raise InferenceError("Invalid input to JSON") from e


@register_request_codec
class JSONRequestCodec(SingleInputRequestCodec):
    InputCodec = JSONCodec
    ContentType = JSONCodec.ContentType