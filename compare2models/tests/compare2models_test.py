import os
import sys
import unittest
from pathlib import Path

from azureml.core import Workspace
from azureml.pipeline.wrapper import Module

sys.path.append(str(Path(__file__).parent.parent))
from compare2models import compare2models


class TestCompare2Models(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Workspace.from_config(str(Path(__file__).parent.parent / 'config.json'))
        cls.base_path = Path(__file__).parent / 'compare2models'

    def prepare_inputs(self) -> dict:
        # Change to your own inputs
        return {
            'first_trained_model': str(self.base_path / 'inputs' / 'first_trained_model'),
            'first_trained_result': str(self.base_path / 'inputs' / 'first_trained_result'),
            'second_trained_model': str(self.base_path / 'inputs' / 'second_trained_model'),
            'second_trained_result': str(self.base_path / 'inputs' / 'second_trained_result')
        }

    def prepare_outputs(self) -> dict:
        # Change to your own outputs
        return {'the_better_model': str(self.base_path / 'outputs' / 'the_better_model')}

    def prepare_parameters(self) -> dict:
        # Change to your own parameters
        return {'str_param': 'some string'}

    def prepare_arguments(self) -> dict:
        # If your input's type is not Path, change this function to your own type.
        result = {}
        result.update(self.prepare_inputs())
        result.update(self.prepare_outputs())
        # result.update(self.prepare_parameters())
        return result

    def test_module_from_func(self):
        # This test calls compare2models from cmd line arguments.
        local_module = Module.from_func(self.workspace, compare2models)
        module = local_module()
        module.set_inputs(**self.prepare_inputs())
        # module.set_parameters(**self.prepare_parameters())
        status = module.run(working_dir=str(self.base_path), use_docker=True)
        self.assertEqual(status, 'Completed', 'Module run failed.')

    def test_module_func(self):
        # This test calls compare2models from parameters directly.
        compare2models(**self.prepare_arguments())
        # check the existence of BestModel
        path = os.path.join(self.prepare_outputs()['the_better_model'], 'BestModel')
        assert os.path.exists(path)
