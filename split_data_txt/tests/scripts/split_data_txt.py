import os
import sys
import random

from azureml.pipeline.wrapper import dsl
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, OutputFile, OutputDirectory, InputFile, InputDirectory


@dsl.module(
    name="Split Data Txt",
    version='0.0.3',
    description='Processing objects: text format data set, each line of the text file is a piece of data, this module divides the data set into training set, verification set and test set.'
)
def split_data_txt(
        training_data_output: OutputFile(type='AnyDirectory'),
        validation_data_output: OutputFile(type='AnyDirectory'),
        test_data_output: OutputFile(type='AnyDirectory'),
        input_dir: InputDirectory(type='AnyDirectory') = None,
        training_data_ratio=0.7,
        validation_data_ratio=0.1,
        random_split=False,
        seed=0
):
    print('============================================')
    print(f"value of input_dir:'{input_dir}', type of input_dir:'{type(input_dir)}'")
    with open(input_dir, 'r', encoding='utf-8') as f:
        data = f.readlines()
    random.seed(seed if random_split else 0)
    # list shuffle
    random.shuffle(data)
    n = len(data)

    training_data_num = int(n * training_data_ratio)
    dev_data_num = int(n * validation_data_ratio)
    train = data[:training_data_num]
    dev = data[training_data_num:training_data_num + dev_data_num]
    test = data[training_data_num + dev_data_num:]
    print('num of total data:', len(data))
    print('num of training data:', len(train))
    print('num of validation data:', len(dev))
    print('num of test_data:', len(test))

    os.makedirs(training_data_output, exist_ok=True)
    path = os.path.join(training_data_output, "train.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(train)

    os.makedirs(validation_data_output, exist_ok=True)
    path = os.path.join(validation_data_output, "dev.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(dev)

    os.makedirs(test_data_output, exist_ok=True)
    path = os.path.join(test_data_output, "test.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(test)
    print('============================================')


if __name__ == '__main__':
    ModuleExecutor(split_data_txt).execute(sys.argv)
