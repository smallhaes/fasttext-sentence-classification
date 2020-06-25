import sys
import unittest
from pathlib import Path

from azureml.core import Workspace
from azureml.pipeline.wrapper import Module

sys.path.append(str(Path(__file__).parent.parent / 'fasttext_predict'))
from fasttext_predict import fasttext_predict


class TestFasttextPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Workspace.from_config(str(Path(__file__).parent.parent / 'config.json'))
        cls.base_path = Path(__file__).parent.parent / 'data'

    def prepare_inputs(self) -> dict:
        # Change to your own inputs
        return {
            'fasttext_model': str(self.base_path / 'fasttext_predict' / 'inputs' / 'fasttext_model'),
            'char2index_dir': str(self.base_path / 'fasttext_predict' / 'inputs' / 'char2index_dir' / 'character2index.json')
        }

    def prepare_outputs(self) -> dict:
        # Change to your own outputs
        return {'output_dir': str(self.base_path / 'fasttext_predict' / 'outputs' / 'output_dir')}

    def prepare_parameters(self) -> dict:
        # Change to your own parameters
        return {'input_sentence': '坚持运动是个好习惯, 不妨每天坚持踢足球'}

    def prepare_arguments(self) -> dict:
        # If your input's type is not Path, change this function to your own type.
        result = {}
        result.update(self.prepare_inputs())
        # result.update(self.prepare_outputs())
        result.update(self.prepare_parameters())
        return result

    def test_module_from_func(self):
        # This test calls fasttext_predict from cmd line arguments.
        local_module = Module.from_func(self.workspace, fasttext_predict)
        module = local_module()
        module.set_inputs(**self.prepare_inputs())
        module.set_parameters(**self.prepare_parameters())
        status = module.run(working_dir=str(self.base_path), use_docker=True)
        self.assertEqual(status, 'Completed', 'Module run failed.')

    def test_module_func(self):
        # This test calls fasttext_predict from parameters directly.
        fasttext_predict(**self.prepare_arguments())
