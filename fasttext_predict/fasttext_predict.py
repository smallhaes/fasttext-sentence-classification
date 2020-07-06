import os
import sys
import torch
from pathlib import Path
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, InputDirectory
from azureml.pipeline.wrapper import dsl
from utils import load_dataset, DataIter, predict


@dsl.module(
    name="FastText Predict",
    version='0.0.20',
    description='Predict the category of the input sentence'
)
def fasttext_predict(
        fasttext_model: InputDirectory(type='AnyDirectory') = '.',
        input_sentence='I like playing football very much',
        char2index_dir: InputDirectory(type='AnyDirectory') = None
):
    print('=====================================================')
    print(f'fasttext_model: {Path(fasttext_model).resolve()}')
    print(f'char2index_dir: {Path(char2index_dir).resolve()}')
    print(f'input_sentence: {input_sentence}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len_ = 38
    path = input_sentence
    test_samples = load_dataset(file_path=path, max_len=max_len_, char2index_dir=char2index_dir)

    test_iter = DataIter(test_samples, batch_size=1)

    path = os.path.join(fasttext_model, 'BestModel')
    model = torch.load(f=path)

    res = predict(model, test_iter, device)
    print('the category of "%s" is %s' % (input_sentence, res))
    print('=====================================================')


if __name__ == '__main__':
    ModuleExecutor(fasttext_predict).execute(sys.argv)
