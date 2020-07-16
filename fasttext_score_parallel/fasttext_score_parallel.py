import os
import sys
import torch
import pandas as pd
from uuid import uuid4
from pathlib import Path
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, InputDirectory, OutputDirectory
from azureml.pipeline.wrapper import dsl
from utils import load_dataset_parallel, DataIter_Parallel, predict_parallel


@dsl.module(
    name="FastText Score Parallel",
    version='0.0.12',
    description='Predict the category of the input sentences',
    job_type='parallel',
    parallel_inputs=[InputDirectory(name='Texts to score')]
)
def fasttext_score_parallel(
        scored_dataset: OutputDirectory(type='AnyDirectory'),
        fasttext_model: InputDirectory(type='AnyDirectory') = '.',
        char2index_dir: InputDirectory(type='AnyDirectory') = None
):
    print('=====================================================')
    print(f'fasttext_model: {Path(fasttext_model).resolve()}')
    print(f'char2index_dir: {Path(char2index_dir).resolve()}')
    print(f'scored_dataset: {scored_dataset}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len_ = 38
    path = os.path.join(fasttext_model, 'BestModel')
    model = torch.load(f=path)

    def run(files):
        if len(files) == 0:
            return []
        print(f"Ready to process {len(files)} texts.")
        print('\n'.join(files))

        with torch.no_grad():
            test_samples = load_dataset_parallel(files=files, max_len=max_len_, char2index_dir=char2index_dir)
            test_iter = DataIter_Parallel(test_samples, shuffle=False)
            results = predict_parallel(model, test_iter, device)
            dict_ = {'Filename': files, 'Class': results}
            df = pd.DataFrame(data=dict_)
            print("Result:")
            print(df)
            output_file = os.path.join(scored_dataset, f"{uuid4().hex}.parquet")
            df.to_parquet(output_file, index=False)
        return results

    return run


if __name__ == '__main__':
    ModuleExecutor(fasttext_score_parallel).execute(sys.argv)
