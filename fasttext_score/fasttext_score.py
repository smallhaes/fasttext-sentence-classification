import os
import sys
import json
import torch
from pathlib import Path
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, InputDirectory, OutputDirectory
from azureml.pipeline.wrapper import dsl
from utils import load_dataset, DataIter, test


@dsl.module(
    name="FastText Score",
    version='0.0.8',
    description='Test the trained FastText model'
)
def fasttext_score(
        model_testing_result: OutputDirectory(type='AnyDirectory'),
        trained_model_dir: InputDirectory(type='AnyDirectory') = None,
        test_data_dir: InputDirectory(type='AnyDirectory') = None,
        char2index_dir: InputDirectory(type='AnyDirectory') = None
):
    print('=====================================================')
    print(f'trained_model_dir: {Path(trained_model_dir).resolve()}')
    print(f'test_data_dir: {Path(test_data_dir).resolve()}')
    print(f'char2index_dir: {Path(char2index_dir).resolve()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len_ = 38
    path = os.path.join(test_data_dir, 'test.txt')
    test_samples = load_dataset(file_path=path, max_len=max_len_, char2index_dir=char2index_dir)

    test_iter = DataIter(test_samples)

    path = os.path.join(trained_model_dir, 'BestModel')
    model = torch.load(f=path)

    path = os.path.join(model_testing_result, 'result.json')
    acc_ = test(model, test_iter, device)
    json.dump({"acc": acc_}, open(path, 'w'))
    print('\n============================================')


if __name__ == '__main__':
    ModuleExecutor(fasttext_score).execute(sys.argv)
