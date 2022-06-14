from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.codecs import NumpyCodec
from mlserver.errors import InferenceError, ModelNotFound
import os
from openvino.runtime import Core
import onnx
from mlserver.utils import get_model_uri
from typing import List

WELLKNOWN_MODEL_FILENAMES = ["model.xml", "model.onnx"]

class OpenvinoRuntime(MLModel):

  async def load(self) -> bool:

    model_uri = await get_model_uri(
      self._settings, wellknown_filenames=WELLKNOWN_MODEL_FILENAMES
    )

    # convert onnx file to openvino format
    if model_uri.endswith('.onnx'):
      model_uri = self.convert_onnx(model_uri)

    # STart model openvino
    self.core = Core()

    if not os.path.exists(model_uri):
      raise ModelNotFound("Cant find model file")

    self._model = self.core.read_model(model=model_uri)
    self.compiled_model_int8 = self.core.compile_model(model=self._model, device_name="CPU")
    self.model_inputs = [list(inp.names)[0] for inp in self.compiled_model_int8.inputs]
    self.model_outputs = [list(inp.names)[0] for inp in self.compiled_model_int8.outputs]

    return await super().load()

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    payload = self._check_request(payload)

    outputs = self._get_model_outputs(payload)

    return InferenceResponse(
      id=payload.id,
      model_name=self.name,
      model_version=self.version,
      outputs=outputs,
    )

  def _get_model_outputs(self, payload: InferenceRequest) -> List[ResponseOutput]:
    for request_output in payload.outputs:
      try:
        X = [self.decode(inp, default_codec=NumpyCodec) for inp in payload.inputs]

        outputs = []

        output_layer_index = self.model_outputs.index(request_output.name)
        y = self.compiled_model_int8(inputs=X)[self.compiled_model_int8.outputs[output_layer_index]]

        output = self.encode(y, request_output, default_codec=NumpyCodec)
        outputs.append(output)
      except ValueError:
        raise InferenceError(f"Cant find output with name {request_output.name}")


    return outputs


  def _check_request(self, payload: InferenceRequest) -> InferenceResponse:
    for inp in payload.inputs:
      if inp.name not in self.model_inputs:
        raise InferenceError(f"No {inp.name} input in model")

    return payload

  @classmethod
  def convert_onnx(cls, file_path):
    '''
    Convert onnx file to openvino format
    :param file_path: str path to onnx file
    :return: None
    '''
    model_onnx = onnx.load(file_path)
    inputs_names = []
    inputs_dim = []
    for inp in model_onnx.graph.input:
      inputs_names.append(inp.name)
      dims = [str(d.dim_value) for d in inp.type.tensor_type.shape.dim]
      dims[0] = "-1"

      inputs_dim.append('[' + ",".join(dims) + ']')

    # Create dir in tmp
    openvino_path = '/tmp/openvino'
    if not os.path.exists(openvino_path):
      os.makedirs(openvino_path)

    # Command
    command = f"mo --input_model \"{file_path}\" --input_shape={','.join(inputs_dim)} --input={','.join(inputs_names)} --output output --output_dir \"{openvino_path}\" --model_name \"onnx_model\""
    os.system(command)

    return os.path.join(openvino_path, 'onnx_model.xml')

