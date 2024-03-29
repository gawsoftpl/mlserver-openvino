from typing import List
import joblib
import cloudpickle
import pickle
import gzip
from mlserver.logging import get_logger
import time

logger = get_logger()

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
        logger.debug(f"Loading transformer {self._file_path}")
        if 'cloudpickle' in self._file_path:
            self._load_cloudpickle()
        elif self._file_path.endswith('pickle'):
            self._load_pickle()
        else:
            self._load_joblib()

    def _load_pickle(self):
        with open(self._file_path,"rb") as f:
            self._pipeline = pickle.load(f)

    def _load_cloudpickle(self):
        if self._file_path.endswith('.gz'):
            with gzip.open(self._file_path, 'rb') as f:
                self._pipeline = cloudpickle.load(f)
        else:
            with open(self._file_path, "rb") as f:
                self._pipeline = cloudpickle.load(f)

    def _load_joblib(self):
        self._pipeline = joblib.load(self._file_path)

    def transform(self, X: List) -> List:
        start_time = time.time()
        X[self._input_index] = self._pipeline.transform(X[self._input_index])
        logger.debug(f"Transform {self._name} elapsed {time.time() - start_time}")
        return X