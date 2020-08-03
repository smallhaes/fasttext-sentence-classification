import os
import sys
import torch
from pathlib import Path
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, InputDirectory
from azureml.pipeline.wrapper import dsl
from inference_schema.schema_decorators import input_schema, output_schema
from utils import load_dataset, DataIter, predict
from azureml.core.model import Model
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType


def init():
    global model
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    model_name = 'BestModel'
    model_path = os.path.join(model_dir, model_name)
    model = torch.load(f=model_path)

standard_sample_input = {'input_sentence': 'i want to travel around the world'}
standard_sample_output = {'category': 'dream'}


@input_schema('param', StandardPythonParameterType(standard_sample_input))
@output_schema(StandardPythonParameterType(standard_sample_output))
def run(param):
    with torch.no_grad():
        path = param['input_sentence']
        max_len_=32
        char2index_dir='deployment/character2index.json'
        test_samples = load_dataset(file_path=path, max_len=max_len_, char2index_dir=char2index_dir)
        test_iter = DataIter(test_samples, batch_size=1)
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            result = predict(model, test_iter, device)
            # You can return any data type, as long as it is JSON serializable.
            return result
        except Exception as e:
            error = str(e)
            return error












