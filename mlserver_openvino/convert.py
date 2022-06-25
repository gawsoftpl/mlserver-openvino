import sys
from model import OpenvinoRuntime

if __name__ == '__main__':
    print("Start converting")
    OpenvinoRuntime.convert_onnx(sys.argv[1], sys.argv[2])