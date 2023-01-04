from typing import Generator, Union, List
import gzip
import base64

PackElement = Union[bytes, str]
PackedPayload = Union[PackElement, List[PackElement]]

def _decompress_gzip_json(body):
    return gzip.decompress(body).decode('utf-8')

def _decompress_base64_gzip_json(body):
    return gzip.decompress(base64.b64decode(body)).decode('utf-8')

def unpack(packed: PackedPayload) -> Generator[PackElement, None, None]:
    if isinstance(packed, list):
        # If it's a list, assume list of strings
        yield from packed
    else:
        # If there is no shape, assume that it's a single element
        yield packed