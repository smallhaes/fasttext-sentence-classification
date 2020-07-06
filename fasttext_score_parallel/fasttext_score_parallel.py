import os
import sys
import json
import torch
from pathlib import Path
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, InputDirectory, OutputDirectory
from azureml.pipeline.wrapper import dsl
from utils import load_dataset, load_dataset_parallel, DataIter, DataIter_Parallel, test
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


@dsl.module(
    name="FastText Score Parallel",
    version='0.0.3',
    description='Test the trained FastText model parallelly',
    job_type='parallel',
    parallel_inputs=[InputDirectory(name='Texts to score')]
)
def fasttext_score_parallel(
        model_testing_result: OutputDirectory(type='AnyDirectory'),
        trained_model_dir: InputDirectory(type='AnyDirectory') = None,
        char2index_dir: InputDirectory(type='AnyDirectory') = None
):
    print('=====================================================')
    print(f'trained_model_dir: {Path(trained_model_dir).resolve()}')
    print(f'char2index_dir: {Path(char2index_dir).resolve()}')

    path = os.path.join(trained_model_dir, 'BestModel')
    model = torch.load(f=path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len_ = 38

    def run(files):
        if len(files) == 0:
            return []
        results = []
        print(f"Ready to process {len(files)} texts.")
        print('\n'.join(files))

        with torch.no_grad():
            test_samples = load_dataset_parallel(files=files, max_len=max_len_, char2index_dir=char2index_dir)
            test_iter = DataIter_Parallel(test_samples)
            acc_ = test(model, test_iter, device)
            path = os.path.join(model_testing_result, 'result.json')
            with open(path, 'w') as f:
                json.dump({"acc": acc_}, f)
            results.append(acc_)
        return results

    return run

def init():
    global run_batch
    run_batch = ModuleExecutor(fasttext_score_parallel).init(sys.argv)


def run(files):
    return run_batch(files)

if __name__ == '__main__':
    ModuleExecutor(fasttext_score_parallel).execute(sys.argv)
