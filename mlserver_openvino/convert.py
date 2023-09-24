import sys
from mlserver_openvino import OpenvinoRuntime
import os
import getopt
import glob

def _convert_onnx_to_openvino(path_in, path_out):
    OpenvinoRuntime.convert_onnx(path_in, path_out)

def convert_onnx_controller(path_in, path_out):
    if os.path.isdir(path_in):
        for file_path in glob.glob(os.path.join(path_in, '**/*.onnx'), recursive=True):
            base_path = os.path.dirname(file_path)
            _convert_onnx_to_openvino(file_path, base_path)
    else:
        _convert_onnx_to_openvino(path_in, path_out)

if __name__ == '__main__':
    options, args = getopt.getopt(sys.argv, "")

    print("Start converting")
    path_in = sys.argv[1]
    path_out = ""
    if len(args) > 2:
        path_out = args[2]

    convert_onnx_controller(path_in, path_out)