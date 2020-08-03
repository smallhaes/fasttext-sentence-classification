# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os
from pathlib import Path
from typing import Union, List

from azureml.pipeline.wrapper import Module
from azureml.pipeline.wrapper.dsl.module import _sanitize_python_variable_name
from azureml.pipeline.wrapper.dsl._module_spec import OutputPort, InputPort, BaseModuleSpec, _get_io_spec_from_module


class _ModuleLocalParamBuilder:
    """Generate default input/output, param and command for module."""

    def __init__(self, input_ports, output_ports, params, source_directory: Path, module_file_name: str):
        self.input_ports = input_ports
        self.output_ports = output_ports
        self.params = params
        # Source directory should be a valid path.
        if not source_directory.exists():
            raise KeyError(f'Source directory {source_directory} does not exist.')
        self.source_folder = source_directory

        self.data_folder = self.source_folder / 'data'
        self.input_dir = self.data_folder / module_file_name / 'inputs'
        self.output_dir = self.data_folder / module_file_name / 'outputs'

        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
        # for vscode launch.json
        self.arguments = []

    def build(self, dry_run=False):
        """Build default input/output, param and command.

        :param dry_run: If specified, won't create input/output.
        """
        # clear existing val
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
        self.arguments = []

        # build input
        self._build_port_argument(self.input_ports, self.input_dir, self.inputs, dry_run)

        # build output
        self._build_port_argument(self.output_ports, self.output_dir, self.outputs, dry_run)

        # build parameters
        for param in self.params:
            key = _sanitize_python_variable_name(param.name)
            if param.default is not None:
                val = param.default
            else:
                val = f'input parameter for {key}'
            self.parameters[key] = val
            self.arguments += [self._to_cli_option_str(param), str(val)]

    def _build_port_argument(self, ports: List[Union[InputPort, OutputPort]],
                             port_dir: Path, port_dict: dict, dry_run=False):
        """Build input/output data

        :param ports: Input/output port.
        :param port_dir: Input/Output directory.
        :param port_dict: Input/Output dictionary.
        :param dry_run: If specified, won't create input/output.
        """
        for port in ports:
            key = _sanitize_python_variable_name(port.name)
            port_path = port_dir / key
            # port value relative to data folder
            value = str(port_path.relative_to(self.data_folder).as_posix())
            port_dict[key] = value
            # arguments relative to source directory
            self.arguments += [self._to_cli_option_str(port),
                               str(port_path.relative_to(self.source_folder).as_posix())]

            if not dry_run:
                if port.type == 'AnyDirectory':
                    os.makedirs(port_path, exist_ok=True)
                elif port.type == 'AnyFile':
                    os.makedirs(str(port_dir), exist_ok=True)
                    open(port_path, 'a').close()
                else:
                    from azureml.pipeline.wrapper.dsl._utils import logger
                    logger.warning('Did not generate file/folder for port {}.'.format(port.name))

    def _to_cli_option_str(self, param):
        if hasattr(param, 'to_cli_option_str') and callable(param.to_cli_option_str):
            return param.to_cli_option_str()
        raise TypeError(f'To build param, param: {param} from spec should have function to_cli_option_str.')


class _ModuleLocalParamBuilderFromSpec(_ModuleLocalParamBuilder):
    def __init__(self, spec: BaseModuleSpec, source_directory: Path, module_file_name: str):
        super(_ModuleLocalParamBuilderFromSpec, self).__init__(
            spec.input_ports, spec.output_ports, spec.params, source_directory, module_file_name)


class _ModuleLocalParamBuilderFromModule(_ModuleLocalParamBuilder):
    def __init__(self, module: Module, source_directory: Path):
        self.module = module
        inputs, outputs, params = _get_io_spec_from_module(module)
        self.module_entry = module._module_dto.entry
        if not self.module_entry.endswith('.py'):
            raise RuntimeError(f'Module entry: {self.module_entry} for module {module.name} did not ends with .py.')
        self.module_file_name = self.module_entry[:-3]

        super().__init__(inputs, outputs, params, source_directory, self.module_file_name)
