import os

import torch
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

from common.utils import load_dataset, DataIter, predict, get_vocab, get_id_label


def init():
    global model, word_to_index, map_label_id
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    path_word_to_index = os.path.join(model_dir, 'word_to_index.json')
    word_to_index = get_vocab(path_word_to_index)
    path_label = os.path.join(model_dir, 'label.txt')
    map_id_label, map_label_id = get_id_label(path_label)
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
        max_len_ = 32
        test_samples = load_dataset(file_path=path, max_len=max_len_, word_to_index=word_to_index,
                                    map_label_id=map_label_id)
        test_iter = DataIter(test_samples, batch_size=1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            result = predict(model, test_iter, device)
            # You can return any data type, as long as it is JSON serializable.
            return result
        except Exception as e:
            error = str(e)
            return error
