import os
import sys
import torch
from pathlib import Path
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, InputDirectory
from azureml.pipeline.wrapper import dsl
from inference_schema.schema_decorators import input_schema, output_schema
from utils import load_dataset, DataIter, predict
# for deployment
from azureml.core.model import Model
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

# standard_sample_input = {'name': ['Sarah', 'John'], 'age': [25, 26]}
# standard_sample_output = {'age': [25, 26]}


standard_sample_input = {'input sentence': 'i want to travel around the world'}
standard_sample_output = {'category': 'dream'}


@input_schema('param', StandardPythonParameterType(standard_sample_input))
@output_schema(StandardPythonParameterType(standard_sample_output))
def fasttext_predict(
        fasttext_model: InputDirectory(type='AnyDirectory') = '.',
        char2index_dir: InputDirectory(type='AnyDirectory') = None
):
    print('=====================================================')
    print(f'fasttext_model: {Path(fasttext_model).resolve()}')
    print(f'char2index_dir: {Path(char2index_dir).resolve()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len_ = 38
    path = os.path.join(fasttext_model, 'BestModel')
    model = torch.load(f=path)

    def run(input_sentence):
        with torch.no_grad():
            path = input_sentence
            test_samples = load_dataset(file_path=path, max_len=max_len_, char2index_dir=char2index_dir)
            test_iter = DataIter(test_samples, batch_size=1)
            try:
                result = predict(model, test_iter, device)
                # You can return any data type, as long as it is JSON serializable.
                return result
            except Exception as e:
                error = str(e)
                return error
    return run


def init():
    global model_for_deployment
    model_for_deployment = ModuleExecutor(fasttext_predict).init(sys.argv)


def run(input_sentence):
    return model_for_deployment(input_sentence)


if __name__ == '__main__':
    ModuleExecutor(fasttext_predict).execute(sys.argv)
