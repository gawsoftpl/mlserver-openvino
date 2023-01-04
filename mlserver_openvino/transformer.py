from typing import List
import joblib

class Transformer:
    '''
    Transform input for specific input index
    '''
    def __init__(self, file_path: str, name: str, input_index: int):
        self._name = name
        self._file_path = file_path
        self._input_index = input_index
        self._pipeline = None

        self._load()

    def _load(self):
        self._pipeline = joblib.load(self._file_path)

    def transform(self, X: List) -> List:
        X[self._input_index] = self._pipeline.transform(X[self._input_index])
        return X