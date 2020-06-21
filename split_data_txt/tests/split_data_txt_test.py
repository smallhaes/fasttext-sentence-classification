import sys
import unittest
from pathlib import Path

from azureml.core import Workspace
from azureml.pipeline.wrapper import Module

sys.path.append(str(Path(__file__).parent.parent))
from split_data_txt import split_data_txt


class TestSplitDataTxt(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Workspace.from_config(str(Path(__file__).parent.parent / 'config.json'))
        cls.base_path = Path(__file__).parent

    def prepare_inputs(self) -> dict:
        # Change to your own inputs
        # return {'input_dir': str(self.base_path / 'inputs' / 'input_dir')}
        return {'input_dir': str(self.base_path / 'inputs' / 'input_dir' / 'THUCNews.txt')}

    def prepare_outputs(self) -> dict:
        # Change to your own outputs
        # return {'output_dir': str(self.base_path / 'outputs' / 'output_dir')}
        return {
            'training_data_output': str(self.base_path / 'outputs' / 'training_data_output'),
            'validation_data_output': str(self.base_path / 'outputs' / 'training_data_output'),
            'test_data_output': str(self.base_path / 'outputs' / 'training_data_output')
        }


    def prepare_parameters(self) -> dict:
        # Change to your own parameters
        return {
            # 'str_param': 'some string',
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
