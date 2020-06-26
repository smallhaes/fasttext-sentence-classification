import os
import sys
import json
import shutil
from pathlib import Path
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, InputDirectory, OutputDirectory
from azureml.pipeline.wrapper import dsl
from azureml.core import Run


@dsl.module(
    name="Compare Two Models",
    version="0.0.9",
    description="Choose the better model according to accuracy"
)
def compare_two_models(
        the_better_model: OutputDirectory(),
        first_trained_model: InputDirectory(type='AnyDirectory') = None,
        first_trained_result: InputDirectory(type='AnyDirectory') = None,
        second_trained_model: InputDirectory(type='AnyDirectory') = None,
        second_trained_result: InputDirectory(type='AnyDirectory') = None,
):
    print('=====================================================')
    print(f'input_dir: {Path(first_trained_model).resolve()}')
    print(f'input_dir: {Path(first_trained_result).resolve()}')
    print(f'input_dir: {Path(second_trained_model).resolve()}')
    print(f'input_dir: {Path(second_trained_result).resolve()}')
    # for logging
    run = Run.get_context()
    path = os.path.join(first_trained_result, 'result.json')
    result_first = json.load(open(path, 'r'))['acc']

    path = os.path.join(second_trained_result, 'result.json')
    second_first = json.load(open(path, 'r'))['acc']

    dst = os.path.join(the_better_model, 'BestModel')
    if result_first >= second_first:
        print('choose the first model')
        run.log(name='which one', value='first')
        src = os.path.join(first_trained_model, 'BestModel')
        shutil.copy(src=src, dst=dst)
    else:
        print('choose the second model')
        run.log(name='which one', value='second')
        src = os.path.join(second_trained_model, 'BestModel')
        shutil.copy(src=src, dst=dst)
    print('=====================================================')


if __name__ == '__main__':
    ModuleExecutor(compare_two_models).execute(sys.argv)
