import os
import sys
import unittest
from pathlib import Path

from azureml.core import Workspace
from azureml.pipeline.wrapper import Module

sys.path.append(str(Path(__file__).parent.parent / 'split_data_txt'))
from split_data_txt import split_data_txt


class TestSplitDataTxt(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Workspace.from_config(str(Path(__file__).parent.parent / 'config.json'))
        cls.base_path = Path(__file__).parent.parent / 'data'

    def prepare_inputs(self) -> dict:
        # Change to your own inputs
        return {'input_dir': str(self.base_path / 'split_data_txt' / 'inputs' / 'input_dir' / 'THUCNews.txt')}

    def prepare_outputs(self) -> dict:
        # Change to your own outputs
        return {
            'training_data_output': str(self.base_path / 'split_data_txt' / 'outputs' / 'training_data_output'),
            'validation_data_output': str(self.base_path / 'split_data_txt' / 'outputs' / 'validation_data_output'),
            'test_data_output': str(self.base_path / 'split_data_txt' / 'outputs' / 'test_data_output'),
        }

    def prepare_parameters(self) -> dict:
        # Change to your own parameters
        return {
            'training_data_ratio': 0.7,
            'validation_data_ratio': 0.1,
            'random_split': False,
            'seed': 123
        }

    def prepare_arguments(self) -> dict:
        # If your input's type is not Path, change this function to your own type.
        result = {}
        result.update(self.prepare_inputs())
        result.update(self.prepare_outputs())
        result.update(self.prepare_parameters())
        return result

    def test_module_from_func(self):
        # This test calls split_data_txt from cmd line arguments.
        local_module = Module.from_func(self.workspace, split_data_txt)
        module = local_module()
        module.set_inputs(**self.prepare_inputs())
        module.set_parameters(**self.prepare_parameters())
        status = module.run(working_dir=str(self.base_path), use_docker=True)
        self.assertEqual(status, 'Completed', 'Module run failed.')

    def test_module_func(self):
        # This test calls split_data_txt from parameters directly.
        split_data_txt(**self.prepare_arguments())
        # check ratio
        params = self.prepare_arguments()
        training_data_ratio = params['training_data_ratio']
        validation_data_ratio = params['validation_data_ratio']
        path_input = self.prepare_inputs()['input_dir']
        num_total = len(open(path_input, 'r', encoding='utf-8').readlines())

        paths = self.prepare_outputs()
        path_training_data_output = os.path.join(paths['training_data_output'], 'train.txt')
        path_validation_data_output = os.path.join(paths['validation_data_output'], 'dev.txt')

        num_train = len(open(path_training_data_output, 'r', encoding='utf-8').readlines())
        num_dev = len(open(path_validation_data_output, 'r', encoding='utf-8').readlines())

        expected_num_train = int(num_total * training_data_ratio)
        expected_num_dev = int(num_total * validation_data_ratio)
        # check ratio
        assert num_train == expected_num_train
        assert num_dev == expected_num_dev
