import os
import sys
import unittest
from pathlib import Path

from azureml.core import Workspace
from azureml.pipeline.wrapper import Module

sys.path.append(str(Path(__file__).parent.parent / 'fasttext_score'))
from fasttext_score import fasttext_score


class TestFasttextScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Workspace.from_config(str(Path(__file__).parent.parent / 'config.json'))
        cls.base_path = Path(__file__).parent.parent / 'data'

    def prepare_inputs(self) -> dict:
        # Change to your own inputs
        return {
            'trained_model_dir': str(self.base_path / 'fasttext_score' / 'inputs' / 'trained_model_dir'),
            'test_data_dir': str(self.base_path / 'fasttext_score' / 'inputs' / 'test_data_dir'),
            'char2index_dir': str(self.base_path / 'fasttext_score' / 'inputs' / 'char2index_dir' / 'character2index.json')
                }

    def prepare_outputs(self) -> dict:
        # Change to your own outputs
        return {'model_testing_result': str(self.base_path / 'fasttext_score' / 'outputs' / 'model_testing_result')}

    def prepare_parameters(self) -> dict:
        # Change to your own parameters
        return {'str_param': ''}

    def prepare_arguments(self) -> dict:
        # If your input's type is not Path, change this function to your own type.
        result = {}
        result.update(self.prepare_inputs())
        result.update(self.prepare_outputs())
        # result.update(self.prepare_parameters())
        return result

    def test_module_from_func(self):
        # This test calls fasttext_score from cmd line arguments.
        local_module = Module.from_func(self.workspace, fasttext_score)
        module = local_module()
        module.set_inputs(**self.prepare_inputs())
        # module.set_parameters(**self.prepare_parameters())
        status = module.run(working_dir=str(self.base_path), use_docker=True)
        self.assertEqual(status, 'Completed', 'Module run failed.')

    def test_module_func(self):
        # This test calls fasttext_score from parameters directly.
        fasttext_score(**self.prepare_arguments())
        # check the existence of result.json
        path_result = os.path.join(self.prepare_outputs()['model_testing_result'], 'result.json')
        assert os.path.exists(path_result)
