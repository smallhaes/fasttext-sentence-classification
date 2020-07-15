# +
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


def init():
    global model
    print('=====================================================')
    fasttext_model = ''
    char2index_dir = ''
    print(f'fasttext_model: {Path(fasttext_model).resolve()}')
    print(f'char2index_dir: {Path(char2index_dir).resolve()}')
    print(f'cur: {os.listdir()}')
    print(f'cur: {os.listdir("deployment")}')
    print(f'cur: {os.listdir("azureml-models")}')
    print(f'cur: {os.listdir("azureml-models/BestModel")}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len_ = 38
    # 'azureml-models\\BestModel\\1\\Trained_model_dir'
#     path = os.path.join('azureml-models','BestModel', '1','Trained_model_dir','BestModel')
    path = os.path.join('azureml-models','BestModel_tmp', '1' ,'BestModel')
    print('======  ')
    print(os.path.exists(path))
    print('isdir',os.path.isdir(path))
    print('isfile',os.path.isfile(path))
    print('isabs',os.path.isabs(path))
    print('islink',os.path.islink(path))
    print('ismount',os.path.ismount(path))
    print('======  ')
    model = torch.load(f=path)

    
'''
standard_sample_input = {'name': ['Sarah', 'John'], 'age': [25, 26]}
standard_sample_output = {'age': [25, 26]}


@input_schema('param', StandardPythonParameterType(standard_sample_input))
@output_schema(StandardPythonParameterType(standard_sample_output))
def standard_py_func(param):
    assert type(param) is dict
    return {'age': param['age']}
'''

standard_sample_input = {'input_sentence': 'i want to travel around the world'}
standard_sample_output = {'category': 'dream'}


@input_schema('param', StandardPythonParameterType(standard_sample_input))
@output_schema(StandardPythonParameterType(standard_sample_output))
def run(param):
    print('===========================')
    print(type(param))
    print('===========================')
    with torch.no_grad():
        path = param['input_sentence']
        print('===========================')
        print('input_sentence: %s'%path)
        print('===========================')
        max_len_=32
        char2index_dir='deployment/character2index.json'
        test_samples = load_dataset(file_path=path, max_len=max_len_, char2index_dir=char2index_dir)
        print('load data succeeded')
        print('===========================')
        test_iter = DataIter(test_samples, batch_size=1)
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            result = predict(model, test_iter, device)
            print('===========================')
            print('===========================')
            # You can return any data type, as long as it is JSON serializable.
            return result
        except Exception as e:
            error = str(e)
            print('===========================')
            print('===========================')
            return error
# -

# standard_sample_input = {'name': ['Sarah', 'John'], 'age': [25, 26]}
# standard_sample_output = {'age': [25, 26]}











