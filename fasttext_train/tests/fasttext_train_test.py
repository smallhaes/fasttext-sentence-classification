import os
import sys
import unittest
from pathlib import Path

from azureml.core import Workspace
from azureml.pipeline.wrapper import Module

sys.path.append(str(Path(__file__).parent.parent))
from fasttext_train import fasttext_train


class TestFasttextTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Workspace.from_config(str(Path(__file__).parent.parent / 'config.json'))
        cls.base_path = Path(__file__).parent / 'fasttext_train'

    def prepare_inputs(self) -> dict:
        # Change to your own inputs
        return {
            'training_data_dir': str(self.base_path / 'inputs' / 'training_data_dir'),
            'validation_data_dir': str(self.base_path / 'inputs' / 'validation_data_dir'),
            'char2index_dir': str(self.base_path / 'inputs' / 'char2index_dir' / 'character2index.json'),
        }

    def prepare_outputs(self) -> dict:
        # Change to your own outputs
        return {'trained_model_dir': str(self.base_path / 'outputs' / 'trained_model_dir')}

    def prepare_parameters(self) -> dict:
        # Change to your own parameters
        return {
            'epochs': 1,
            'batch_size': 64,
            'learning_rate': 0.0005,
            'embedding_dim': 128
        }

    def prepare_arguments(self) -> dict:
        # If your input's type is not Path, change this function to your own type.
        result = {}
        result.update(self.prepare_inputs())
        result.update(self.prepare_outputs())
        result.update(self.prepare_parameters())
        return result

    def test_module_from_func(self):
        # This test calls fasttext_train from cmd line arguments.
        local_module = Module.from_func(self.workspace, fasttext_train)
        module = local_module()
        module.set_inputs(**self.prepare_inputs())
        module.set_parameters(**self.prepare_parameters())
        status = module.run(working_dir=str(self.base_path), use_docker=True)
        self.assertEqual(status, 'Completed', 'Module run failed.')

    def test_module_func(self):
        # This test calls fasttext_train from parameters directly.
        fasttext_train(**self.prepare_arguments())
        # check the existence of BestModel
        path_model = os.path.join(self.prepare_outputs()['trained_model_dir'], 'BestModel')
        assert os.path.exists(path_model)
