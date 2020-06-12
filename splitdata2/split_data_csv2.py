import sys
import os

from azureml.pipeline.wrapper import dsl
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, OutputFile, OutputDirectory, InputFile, InputDirectory
import pandas as pd

@dsl.module(
    job_type="basic",
    name="split_data_csv2",
    version='0.0.2'
)
def split_data_csv2(
        training_data_output: OutputFile(type='AnyDirectory'),
        validation_data_output: OutputFile(type='AnyDirectory'),
        test_data_output: OutputFile(type='AnyDirectory'),
        input_dir: InputDirectory(type='AnyDirectory') = None,  # 在designer为str类型
        # separator='\t',
        training_data_ratio=0.7,
        validation_data_ratio=0.1,
        random_split=False,
        seed=0
):
    """A sample module use different parameter types and customized input/output ports."""
    print(f"input_dir的值为:'{input_dir}', input_dir的类型为:'{type(input_dir)}'")
    separator='\t'
    df = pd.read_csv(input_dir, sep=separator)
    # pandas打乱行
    df = df.sample(frac=1, random_state=seed if random_split else 0)
    df.reset_index(drop=True, inplace=True)

    data_num = df.shape[0]
    training_data_num = int(data_num * training_data_ratio)
    validation_data_num = int(data_num * validation_data_ratio)
    training_data = df.iloc[:training_data_num, :]
    validation_data = df.iloc[training_data_num:training_data_num + validation_data_num, :]
    test_data = df.iloc[training_data_num + validation_data_num:, :]
    print('training_data数量:', training_data.shape[0])
    print('validation_data数量:', validation_data.shape[0])
    print('test_data数量:', test_data.shape[0])

    os.makedirs(training_data_output, exist_ok=True)
    path = os.path.join(training_data_output, "training_data.csv")
    training_data.to_csv(path_or_buf=path, index=False, sep=separator)

    os.makedirs(validation_data_output, exist_ok=True)
    path = os.path.join(validation_data_output, "validation_data.csv")
    validation_data.to_csv(path_or_buf=path, index=False, sep=separator)

    os.makedirs(test_data_output, exist_ok=True)
    path = os.path.join(test_data_output, "test_data.csv")
    test_data.to_csv(path_or_buf=path, index=False, sep=separator)

if __name__ == "__main__":
    ModuleExecutor(split_data_csv2).execute(sys.argv)
