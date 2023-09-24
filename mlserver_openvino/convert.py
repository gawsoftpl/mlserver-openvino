import sys
from mlserver_openvino import OpenvinoRuntime
import os
import getopt

def _convert_onnx_to_openvino(path_in, path_out):
    OpenvinoRuntime.convert_onnx(path_in, path_out)

def convert_onnx_controller(path_in, path_out):
    if os.path.isdir(path_in):
        abs_path = os.path.abspath(path_in)
        save_path = os.path.dirname(abs_path)
        for file_path in glob.glob(os.path.join(path_in, '**/*.onnx'), recursive=True):
            _convert_onnx_to_openvino(file_path, save_path)
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