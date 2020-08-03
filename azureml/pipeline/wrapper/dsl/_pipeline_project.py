# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""A helper class which builds a pipeline project skeleton."""
import argparse
import inspect
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Union, List
import ruamel.yaml as yaml

from azureml.pipeline.wrapper.dsl._module_local_param_builder import _ModuleLocalParamBuilderFromSpec
from azureml.pipeline.wrapper.dsl._argparse import gen_module_by_argparse
from azureml.pipeline.wrapper.dsl._module_generator import is_py_file
from azureml.pipeline.wrapper.dsl._version import VERSION
from azureml.pipeline.wrapper.dsl._utils import _sanitize_python_class_name, logger, BackUpFiles, _log_file_skip, \
    _log_file_update, _log_file_create, _log_without_dash, to_literal_str, _find_py_files_in_target, \
    _split_path_into_list, _change_working_dir, _is_function, _get_func_def, _import_module_with_working_dir, \
    _has_dsl_module_str, _get_module_path_and_name_from_source, _relative_to, timer_decorator, _source_exists
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, _sanitize_python_variable_name, TooManyDSLModulesError
from azureml.pipeline.wrapper.dsl._module_spec import SPEC_EXT

# TODO: make this a param of _ModuleResourceBase
DATA_PATH = Path(__file__).resolve().parent / 'data'
MODULE_NAME_CONST = 'MODULE_NAME'
MODULE_ENTRY_CONST = 'MODULE_ENTRY'
FUNCTION_NAME_CONST = 'FUNCTION_NAME'
MODULE_CLASS_NAME_CONST = 'MODULE_CLASS_NAME'
PIPELINE_NAME_CONST = 'PIPELINE_NAME'
EXPERIMENT_NAME = 'EXPERIMENT_NAME'
DSL_PARAM_DICT_CONST = "'DSL_PARAM_DICT'"


def get_template_file(template_name, job_type='basic'):
    path = DATA_PATH / '{}_module'.format(job_type) / template_name
    if path.is_file():
        return path
    if job_type == 'basic':
        raise FileNotFoundError("The template of basic module is not found: %r" % path)
    # If job_type specific template file doesn't exist, return the default template of basic module.
    return get_template_file(template_name)


class _ModuleObject:
    """Wraps module executor and provides params according to it."""

    def __init__(self, module: ModuleExecutor):
        # TODO: refine this class provide template needed values
        self.module_executor = module
        self.spec = module.spec

        # code path: absolute path of module entry
        self.module_entry_path = Path(inspect.getfile(module._func)).absolute()
        # source directory is current folder
        self.source_directory = Path(os.getcwd())
        # module entry: source dir -> module entry func
        module_entry = str(_relative_to(self.module_entry_path, self.source_directory, raises_if_impossible=True))
        if module_entry.endswith('.py'):
            module_entry = module_entry[:-3]
        self.module_entry = '.'.join(_split_path_into_list(module_entry))

        # use file name as built resource name prefix
        self.sanitized_entry_name = _sanitize_python_variable_name(self.module_entry_path.stem)
        self.sanitized_module_name = _sanitize_python_variable_name(module.name)
        self.function_name = module._func.__name__

        # generate default input/output, param and command.
        self.module_param_builder = _ModuleLocalParamBuilderFromSpec(
            module.spec, self.source_directory, self.module_entry_path.stem)
        self.module_param_builder.build(dry_run=True)


class _ModuleResourceBase:
    def __init__(self, folder: Path, path, type, template, source_dir=None):
        if source_dir is None:
            source_dir = folder
        self.folder = folder
        self.path = folder / path
        self.type = type
        self.template = template
        self.source_directory = source_dir

    @property
    def path(self):
        return Path(self._path)

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def file_info(self):
        return _relative_to(self.path, self.source_directory, raises_if_impossible=True)

    def create(self):
        # default behavior: copy template to path
        shutil.copyfile(self.template, self.path)

    def update(self):
        # call create if file not exist, perform according to strategy if exist.
        pass

    def skip(self):
        _log_file_skip(self.file_info)


class _ModuleResourceWithBackupStrategy(_ModuleResourceBase):
    """Update strategy: Back up resource into backup folder."""

    def __init__(self, folder, path, type, template, backup_folder: Path, source_dir=None):
        super().__init__(folder, path, type, template, source_dir)
        self._backup_folder = backup_folder

    def update(self):
        # back up original file when trying to update it.
        file_exist = self.path.exists()
        if file_exist:
            relative_path = _relative_to(self.path, self.folder, raises_if_impossible=True)
            os.makedirs((self._backup_folder / relative_path).parent, exist_ok=True)
            shutil.copy(self.path, self._backup_folder / relative_path)
        result = self.create()
        if file_exist:
            _log_file_update(self.file_info)
        else:
            _log_file_create(self.file_info)
        return result


class _ModuleResourceWithPreserveStrategy(_ModuleResourceBase):
    """Update strategy: won't update if file exists."""

    def update(self):
        # preserve the original file when trying to update it.
        if self.path.exists():
            self.skip()
        else:
            self.create()
            _log_file_create(self.file_info)


class _ModuleResourceWithExceptionStrategy(_ModuleResourceBase):
    """Update strategy: throw exception if file exists."""

    def update(self):
        # raise exception when trying to trying to update existing file.
        if self.path.exists():
            raise FileExistsError("Target file '{}' already exists.".format(self.path))
        else:
            self.create()
            _log_file_create(self.file_info)


class _SpecFile(_ModuleResourceWithBackupStrategy):
    def __init__(self, module: _ModuleObject, folder: Path, backup_folder, source_dir, spec_path=None):
        self.module_object = module
        if spec_path is None:
            spec_path = self.module_object.sanitized_entry_name + SPEC_EXT
        super().__init__(folder, spec_path, self.__class__.__name__, None, backup_folder, source_dir)

    @property
    def file_info(self):
        return '{} -> {}'.format(
            _relative_to(self.module_object.module_entry_path, self.source_directory, raises_if_impossible=True),
            _relative_to(self.path, self.source_directory, raises_if_impossible=True))

    def create(self):
        os.makedirs(self.path.parent, exist_ok=True)
        self.module_object.module_executor.to_spec_yaml(folder=self.folder, spec_file=self.path.name)


class _NotebookFile(_ModuleResourceWithPreserveStrategy):
    NOTEBOOK_TEMPLATE_NAME = 'notebook_sample_code.template'

    def __init__(self, module: _ModuleObject, folder: Path, job_type='basic'):
        self.module_object = module
        path = '{}_test.ipynb'.format(module.sanitized_entry_name)
        self.notebook_template = get_template_file(self.NOTEBOOK_TEMPLATE_NAME, job_type)
        self.job_type = job_type
        super().__init__(folder, path, self.__class__.__name__, self.notebook_template)

    def create(self):
        os.makedirs(self.path.parent, exist_ok=True)
        with open(self.template) as file:
            notebook_content = file.read()
            notebook_content = notebook_content.replace(
                MODULE_ENTRY_CONST, self.module_object.module_entry).replace(
                FUNCTION_NAME_CONST, self.module_object.function_name).replace(
                MODULE_NAME_CONST, self.module_object.sanitized_module_name).replace(
                EXPERIMENT_NAME, '{}_experiment'.format(self.module_object.sanitized_module_name)).replace(
                PIPELINE_NAME_CONST, '{}_pipeline'.format(self.module_object.sanitized_module_name))

            notebook_content = self._update_notebook_by_job_type(notebook_content)

            with open(self.path, 'w') as out_file:
                out_file.write(notebook_content)

    def _update_notebook_by_job_type(self, notebook_content):
        # This is a hard-coded logic to make sure runsetting is correctly set for different job type.
        # TODO: Refine the update logic here.
        target = '' if self.job_type != 'parallel' else \
            "module1.runsettings.configure(node_count=1, process_count_per_node=1)"
        return notebook_content.replace("module1.runsettings.configure()", target)


class _UnittestFile(_ModuleResourceWithPreserveStrategy):
    INPUTS_CONST = 'INPUTS_TEMPLATE'
    OUTPUTS_CONST = 'OUTPUTS_TEMPLATE'
    PARAMETERS_CONST = 'PARAMETERS_TEMPLATE'
    UT_TEMPLATE_NAME = 'unittest_sample_code.template'

    def __init__(self, module: _ModuleObject, folder: Path, job_type='basic'):
        self.module_object = module
        path = '{}_test.py'.format(module.sanitized_entry_name)
        self.ut_template = get_template_file(self.UT_TEMPLATE_NAME, job_type)
        super().__init__(folder, path, self.__class__.__name__, self.ut_template)

    def create(self):
        os.makedirs(self.path.parent, exist_ok=True)
        with open(self.template) as file:
            ut_code = file.read()
            module_class_name = _sanitize_python_class_name(self.module_object.sanitized_entry_name)
            ut_code = ut_code.replace(
                MODULE_ENTRY_CONST, self.module_object.module_entry).replace(
                FUNCTION_NAME_CONST, self.module_object.function_name).replace(
                MODULE_NAME_CONST, self.module_object.sanitized_module_name).replace(
                MODULE_CLASS_NAME_CONST, module_class_name).replace(
                self.INPUTS_CONST,
                self._to_absolute_path_literal(self.module_object.module_param_builder.inputs)).replace(
                self.OUTPUTS_CONST,
                self._to_absolute_path_literal(self.module_object.module_param_builder.outputs)).replace(
                self.PARAMETERS_CONST, str(self.module_object.module_param_builder.parameters))

            # use docker=False for test env
            if '_TEST_ENV' in os.environ:
                ut_code = ut_code.replace('use_docker=True', 'use_docker=False')

        with open(self.path, 'w') as out_file:
            out_file.write(ut_code)

    @classmethod
    def _to_absolute_path_literal(cls, data: dict):
        # path in data should be posix format
        literal_items = []
        for key, value in data.items():
            path_parts = ["'{}'".format(part) for part in value.split('/')]
            path_parts_str = ' / '.join(path_parts)
            literal_items.append(
                "'{}': str(self.base_path / {})".format(key, path_parts_str))

        return '{' + ','.join(literal_items) + '}'


class _InitFile(_ModuleResourceWithPreserveStrategy):
    def __init__(self, folder):
        path = '__init__.py'
        super().__init__(folder, path, self.__class__.__name__, None)

    def create(self):
        open(self.path, 'a').close()


class _VSCodeFile(_ModuleResourceWithPreserveStrategy):
    VSCODE_DIR = '.vscode'


class _VSCodeLaunch(_VSCodeFile):
    VSCODE_LAUNCH_CONFIG = 'launch.json'
    VSCODE_LAUNCH_CONFIG_TEMPLATE = DATA_PATH / 'inputs' / VSCODE_LAUNCH_CONFIG

    def __init__(self, folder: Path, arguments: list):
        super().__init__(folder, os.path.join(self.VSCODE_DIR, self.VSCODE_LAUNCH_CONFIG), self.__class__.__name__,
                         self.VSCODE_LAUNCH_CONFIG_TEMPLATE)
        # arguments: list of  program -> args dict
        configurations = []
        for argument in arguments:
            args = []
            for arg in argument.get('args', []):
                # corner case in windows: empty string couldn't pass to program in vscode, escape it
                if arg == "" and platform.system() == 'Windows':
                    args.append("\"\"")
                else:
                    args.append(arg)
            name = argument.get('name', None)
            program = argument.get('program', None)
            configurations.append({
                "name": name,
                "type": "python",
                "request": "launch",
                "args": args,
                "console": "integratedTerminal",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                },
                "program": str((Path('${workspaceFolder}') / program).as_posix())
            })
        self.configurations = configurations

    def create(self):
        with open(self.template) as file:
            data = json.load(file)
            data['configurations'] = self.configurations
            with open(self.path, 'w') as out_file:
                json.dump(data, out_file, indent=4)


class _VSCodeSetting(_VSCodeFile):
    VSCODE_SETTINGS_CONFIG = 'settings.json'
    VSCODE_SETTINGS_CONFIG_TEMPLATE = DATA_PATH / 'inputs' / VSCODE_SETTINGS_CONFIG

    def __init__(self, folder: Path):
        super().__init__(folder, os.path.join(self.VSCODE_DIR, self.VSCODE_SETTINGS_CONFIG), self.__class__.__name__,
                         self.VSCODE_SETTINGS_CONFIG_TEMPLATE)


class _GitIgnore(_ModuleResourceWithPreserveStrategy):
    GIT_IGNORE = '.gitignore'
    GIT_IGNORE_TEMPLATE = DATA_PATH / GIT_IGNORE

    def __init__(self, folder: Path):
        super().__init__(folder, self.GIT_IGNORE, self.__class__.__name__, self.GIT_IGNORE_TEMPLATE)


class _AMLIgnore(_ModuleResourceWithPreserveStrategy):
    AML_IGNORE = '.amlignore'
    AML_IGNORE_TEMPLATE = DATA_PATH / AML_IGNORE

    def __init__(self, folder: Path):
        super().__init__(folder, self.AML_IGNORE, self.__class__.__name__, self.AML_IGNORE_TEMPLATE)


class _WorkspaceConfig(_ModuleResourceWithPreserveStrategy):
    WORKSPACE_CONFIG = 'config.json'
    WORKSPACE_CONFIG_TEMPLATE = DATA_PATH / WORKSPACE_CONFIG

    def __init__(self, folder: Path):
        super().__init__(folder, self.WORKSPACE_CONFIG, self.__class__.__name__, self.WORKSPACE_CONFIG_TEMPLATE)


class _BasicModuleEntryFromTemplate(_ModuleResourceWithExceptionStrategy):
    CODE_TEMPLATE = DATA_PATH / 'basic_module' / 'basic_module.template'

    def __init__(self, folder: Path, name):
        self.module_name = name
        self.sanitized_name = _sanitize_python_variable_name(name)
        path = '{}.py'.format(self.sanitized_name)
        super().__init__(folder, path, self.__class__.__name__, self.CODE_TEMPLATE)

    def create(self):
        dsl_param_dict = {'name': self.module_name}
        dsl_param_dict_str = ',\n    '.join(
            ['%s=%s' % (key, to_literal_str(value)) for key, value in dsl_param_dict.items()])
        os.makedirs(self.path.parent, exist_ok=True)
        with open(self.template) as file:
            sample_code = file.read()
            sample_code = sample_code. \
                replace(DSL_PARAM_DICT_CONST, dsl_param_dict_str). \
                replace(MODULE_NAME_CONST, self.sanitized_name)

        with open(self.path, 'w') as out_file:
            out_file.write(sample_code)

    @classmethod
    def entry_from_type(cls, job_type, name):
        job_type = job_type.lower().strip()
        type_to_entry_class = {
            'basic': _BasicModuleEntryFromTemplate,
            'mpi': _MpiModuleEntryFromTemplate,
            'parallel': _ParallelModuleEntryFromTemplate,
        }
        if job_type not in type_to_entry_class:
            raise RuntimeError('Job type: %r not supported.' % job_type)
        return type_to_entry_class[job_type](Path(os.getcwd()), name)


class _MpiModuleEntryFromTemplate(_BasicModuleEntryFromTemplate):
    CODE_TEMPLATE = DATA_PATH / 'mpi_module' / 'mpi_module.template'


class _ParallelModuleEntryFromTemplate(_BasicModuleEntryFromTemplate):
    CODE_TEMPLATE = DATA_PATH / 'parallel_module' / 'parallel_module.template'


class _BasicModuleEntryFromFunction(_ModuleResourceWithExceptionStrategy):
    CODE_TEMPLATE = DATA_PATH / 'function_sample_code.template'

    def __init__(self, folder: Path, name, function):
        self.module_path, self.func_name = '.'.join(function.split('.')[:-1]), function.split('.')[-1]
        if name is None:
            name = self.func_name
        else:
            name = _sanitize_python_variable_name(name)

        if self.module_path == '':
            raise ValueError(
                "Invalid function: %s, please make sure the format is 'some_module.func_name'" % function)
        try:
            module = _import_module_with_working_dir(self.module_path, str(folder))
            self.func = getattr(module, self.func_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "Import function '%s' failed at target folder '%s'\n" % (function, os.getcwd())) from e
        # gen module entry to function's folder
        entry_folder = Path(inspect.getfile(module)).parent
        path = '{}.py'.format(name)
        super().__init__(entry_folder, path, self.__class__.__name__, self.CODE_TEMPLATE, folder)

    def create(self):
        with open(self.template) as fin:
            code = fin.read()
            code = code.replace(MODULE_NAME_CONST, self.module_path). \
                replace('FUNC_NAME', self.func_name). \
                replace('FUNC_DEF', _get_func_def(self.func))
        with open(self.path, 'w') as fout:
            fout.write(code)
        io_hint = "Please use InputDirectory/InputFile/OutputDirectory/OutputFile" + \
                  " in function annotation to hint inputs/outputs."
        logger.info(('Generated entry file: %s.\n' + io_hint) % self.path)


class _PipelineProjectProperties(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update({
            'NotebookFolder': 'notebooks',
            'TestsFolder': 'tests',
            'DataFolder': 'data'
        })

    @property
    def entry_folder(self):
        return self.get('EntryFolder', 'entries')

    @property
    def spec_folder(self):
        return self.get('SpecFolder', 'specs')

    @property
    def notebook_folder(self):
        return self.get('NotebookFolder', 'notebooks')

    @property
    def tests_folder(self):
        return self.get('TestsFolder', 'tests')

    @property
    def data_folder(self):
        return self.get('DataFolder', 'data')


class _VSCodeResource:
    def __init__(self, folder: Path, arguments: list):
        self.folder = folder
        self.launch = _VSCodeLaunch(folder, arguments)
        self.setting = _VSCodeSetting(folder)

    def init(self):
        logger.info('Initializing vscode settings...')
        os.makedirs(str(self.folder / _VSCodeFile.VSCODE_DIR), exist_ok=True)
        self.launch.update()
        self.setting.update()


class _ModuleResource:
    MODULE_ENTRY_FILE_CONST = 'entry'
    SOURCE_DIR_CONST = 'sourceDirectory'

    def __init__(self, module: _ModuleObject, backup_folder, properties=None, spec_path=None, job_type='basic'):
        if properties is None:
            properties = _PipelineProjectProperties()

        self.source_directory = module.source_directory
        self.test_folder = self.source_directory / properties.tests_folder
        self.module_object = module
        self.module_name = module.module_executor.name

        # spec will be generated under module entry folder.
        self.spec_file = _SpecFile(module, module.module_entry_path.parent, backup_folder, self.source_directory,
                                   spec_path)
        self.init_resources = [
            _NotebookFile(module, self.source_directory, job_type),
            _UnittestFile(module, self.test_folder, job_type),
            _GitIgnore(self.source_directory),
            _AMLIgnore(self.source_directory),
            _WorkspaceConfig(self.source_directory)
        ]

        argument = {'name': str(self.module_object.module_entry_path.name),
                    'args': self.module_object.module_param_builder.arguments,
                    'program': str(
                        _relative_to(self.module_object.module_entry_path, self.source_directory,
                                     raises_if_impossible=True))}
        self.vscode_resource = _VSCodeResource(self.source_directory, [argument])

    @property
    def dict(self) -> dict:
        # all path here should be relative path
        entry_path = str(_relative_to(self.module_object.module_entry_path, os.getcwd()).as_posix())
        source_dir = str(_relative_to(self.source_directory, os.getcwd()).as_posix())
        result = {
            self.MODULE_ENTRY_FILE_CONST: entry_path,
            self.SOURCE_DIR_CONST: source_dir
        }
        return result

    def init(self):
        self.create_data()
        self.build()

        for resource in self.init_resources:
            resource.update()

        self.vscode_resource.init()

    def build(self):
        # Build spec in source directory
        self.spec_file.update()

    def create_data(self):
        self.module_object.module_param_builder.build()

    @staticmethod
    def _create_entry_folder(entry_folder):
        os.makedirs(entry_folder, exist_ok=True)
        while entry_folder != Path('.'):
            init_file = _InitFile(entry_folder)
            init_file.update()
            entry_folder = entry_folder.parent

    @staticmethod
    def collect_module_from_source(source, raise_import_error, raise_other_errors=False, working_dir=None) -> \
            Union[None, ModuleExecutor]:
        if working_dir is None:
            working_dir = os.getcwd()
        try:
            # Only treat source as file if is's a python file. Otherwise, treat it as a module and let importlib help.
            # Force reload in case when module with same name in .moduleproj
            if is_py_file(source):
                module = ModuleExecutor.collect_module_from_file(source, working_dir)
            else:
                module = ModuleExecutor.collect_module_from_py_module(source, working_dir)
            return module
        except ImportError as e:
            logger.error(
                "Failed to resolve path: {}, "
                "please make sure all requirements inside conda.yaml has been installed. \n"
                "Error message: {}".format(source, str(e)))
            if raise_import_error:
                raise e
        except BaseException as e:
            logger.error(
                'Failed to resolve path: {}. \n'
                'Error message: {}'.format(source, str(e))
            )
            if raise_other_errors:
                raise e

    @staticmethod
    def from_dsl_module_file(
            path, backup_folder,
            job_type='basic', raise_import_error=False,
    ) -> Union['_ModuleResource', None]:
        """Create module object from file."""
        module = _ModuleResource.collect_module_from_source(str(path), raise_import_error)
        if module is None:
            return None
        module_object = _ModuleObject(module)
        return _ModuleResource(module_object, backup_folder=backup_folder, job_type=job_type)

    @classmethod
    def load_from_config(cls, module_config, backup_folder):
        """Load module resource from config.

        :param module_config: Config dict.
        :param backup_folder: Backup folder
        :return: module entry path, loaded module resource
        """
        module_path = module_config[cls.MODULE_ENTRY_FILE_CONST]
        source_dir = module_config.get(cls.SOURCE_DIR_CONST, '.')
        if not Path(source_dir).exists():
            raise ValueError(f'{cls.SOURCE_DIR_CONST}: {source_dir} not exist.')
        if module_path is None:
            return None, None
        with _change_working_dir(source_dir):
            # assumes all module file in module project is dsl.module
            module_resource = _ModuleResource.from_dsl_module_file(module_path, backup_folder)
        return module_path, module_resource


class PipelineProject:
    """A helper class which builds a pipeline project skeleton."""

    # TODO: align with file name and class name
    MODULE_PROJECT_FILE_NAME = '.moduleproj'
    MODULES_KEY = 'modules'

    def __init__(self, modules: List[_ModuleResource], properties=None):
        """Build a pipeline project skeleton."""
        self.modules = modules
        if properties is None:
            properties = _PipelineProjectProperties()
        self.properties = properties

    @property
    def dict(self):
        """Transform a pipeline project to dictionary."""
        return {
            self.MODULES_KEY: [module.dict for module in self.modules],
        }

    def dump(self):
        """Dump a pipeline project into file."""
        if len(self.modules) == 0:
            return
        else:
            code_folder = self.modules[0].source_directory
            with _change_working_dir(code_folder):
                logger.info('Dumping configurations into {}'.format(Path(self.MODULE_PROJECT_FILE_NAME).absolute()))
                if Path(self.MODULE_PROJECT_FILE_NAME).exists():
                    existing_project, _ = PipelineProject.load_from_project_file(None)
                    PipelineProject.merge(self, existing_project)
                with open(self.MODULE_PROJECT_FILE_NAME, 'w') as file:
                    yaml.dump(self.dict, file, indent=4, default_flow_style=False)

    @staticmethod
    def version_hint():
        """Show the version info of current ModuleProject and python environment."""
        return "Module project builder version: %s Python executable: %s" % (VERSION, sys.executable)

    @staticmethod
    @timer_decorator
    def init(source=None, name=None, job_type='basic', source_dir=None, inputs=None, outputs=None):
        """Init a Pipeline project skeleton which contains module file, module spec and jupyter notebook.

        :param source: Source for specific mode, could be pacakge.function or path/to/python_file.py
        :param name: The name of module. eg: Select Columns.
        :param job_type: Job type of the module. Could be basic, mpi, hdinsight, parallel.
            Defaults to basic if not specified, which refers to run job on a single compute node.
        :param source_dir: Source directory.
        :return: Created module project.
        :rtype: ModuleProject
        """
        logger.info(PipelineProject.version_hint())
        if source_dir is None:
            source_dir = os.getcwd()
        if source is None:
            # When initiating from template, source dir is seperate folder.
            if name is not None:
                source_dir = Path(source_dir) / _sanitize_python_variable_name(name)
            else:
                raise KeyError('Name and source can not be empty at the same time.')
        with _change_working_dir(source_dir), BackUpFiles(os.getcwd()) as backup_folder:
            _log_without_dash('========== Init started: {} =========='.format(os.getcwd()))

            resource = PipelineProject.collect_or_gen_dsl_module(
                source, name, job_type, backup_folder,
                inputs=inputs, outputs=outputs)
            resources = []
            if resource is not None:
                resources.append(resource)
            pipeline_project = PipelineProject(resources)
            for module in pipeline_project.modules:
                logger.info('Initializing {}...'.format(module.module_name))
                module.init()

            _log_without_dash('========== Init succeeded =========='.format(len(pipeline_project.modules)))
            pipeline_project.dump()
            return pipeline_project

    @staticmethod
    @timer_decorator
    def build(target=None, source_dir=None):
        """Build module spec for dsl.module.

        :param target: could be a dsl.module entry file or folder, will be os.getcwd() if not set.
        :param source_dir: Source directory.
        """
        logger.info(PipelineProject.version_hint())
        if source_dir is None:
            source_dir = os.getcwd()
        source_dir = Path(source_dir).absolute()

        with _change_working_dir(source_dir), BackUpFiles(source_dir) as backup_folder:
            if target is None:
                target = os.getcwd()
            target = Path(target)

            # Check if target is valid file/dir when loading it.
            if target.is_absolute():
                try:
                    target = _relative_to(target, source_dir, raises_if_impossible=True)
                except ValueError as e:
                    raise ValueError(
                        'Target should be inside source directory. Got {} and {} instead.'.format(
                            target, source_dir)) from e

            if target.is_file and target.name == PipelineProject.MODULE_PROJECT_FILE_NAME:
                # If a .pipelineproj file is passed, target is it's parent.
                target = target.parent
            _log_without_dash('========== Build started: {} =========='.format(os.getcwd()))
            pipeline_project, failed_modules = PipelineProject.load(target, backup_folder)
            for module_resource in pipeline_project.modules:
                logger.info(
                    'Building module:{} into spec in source directory: {}...'.format(
                        module_resource.module_name, module_resource.source_directory))
                module_resource.build()
            _log_without_dash(
                '========== Build: {} succeeded, {} failed =========='.format(len(pipeline_project.modules),
                                                                              len(failed_modules)))
            return pipeline_project

    @staticmethod
    def collect_or_gen_dsl_module(
            source, name, job_type, backup_folder,
            inputs=None, outputs=None,
    ) -> Union[_ModuleResource]:
        """ Collect or generate _ModuleResource, raises exception if multi dsl.modules are collected."""
        # collect
        if source is not None and _has_dsl_module_str(source):
            # Skip import if no dsl module str in module to prevent potential import errors.
            logger.info('Attempting to load dsl.modules from source...')
            try:
                module = _ModuleResource.collect_module_from_source(source, raise_import_error=False,
                                                                    raise_other_errors=True)
                if module is not None:
                    module_object = _ModuleObject(module)
                    return _ModuleResource(module_object, backup_folder=backup_folder)
            except TooManyDSLModulesError as e:
                raise e
            except BaseException:
                pass

        # generate
        if source is None:
            logger.info('Attempting to generate dsl.module from template...')
            file = PipelineProject.gen_dsl_module_from_name(name, job_type)
        elif _is_function(source):
            logger.info('Attempting to generate dsl.module from function...')
            if job_type != 'basic':
                raise KeyError('Unsupported job-type {} when init from {}'.format(job_type, source))
            file = PipelineProject.gen_dsl_module_from_func(name, source)
        else:
            logger.info('Attempting to generate dsl.module from arg parser...')
            if not _source_exists(source):
                raise KeyError('Source: {} does not exist as file.'.format(source))
            file = PipelineProject.gen_dsl_module_from_argparse(name, job_type, source, inputs=inputs, outputs=outputs)

        module_resource = _ModuleResource.from_dsl_module_file(str(file), backup_folder, job_type=job_type)
        if module_resource is None:
            raise RuntimeError('Failed to resolve generated dsl.module file: {}'.format(file))
        return module_resource

    @staticmethod
    def gen_dsl_module_from_name(name, job_type):
        """Generate a start up dsl.module file."""
        code_template = _BasicModuleEntryFromTemplate.entry_from_type(job_type=job_type, name=name)
        code_template.update()
        return code_template.path

    @staticmethod
    def gen_dsl_module_from_func(name, function):
        """Generate a dsl.module file from function."""
        code_template = _BasicModuleEntryFromFunction(Path(os.getcwd()), name, function)
        code_template.update()
        return code_template.path

    @staticmethod
    def gen_dsl_module_from_argparse(name, job_type, source, inputs=None, outputs=None):
        """Generate a dsl.module from arg parse."""
        module_path, module_name = _get_module_path_and_name_from_source(source)
        entry_path = str(Path(module_path) / '{}_entry.py'.format(module_name))
        gen_module_by_argparse(
            entry=source, target_file=entry_path, inputs=inputs, outputs=outputs,
            # Here we hard-coded set job_type=None if it is 'basic'
            # because 'basic' is the default value even we don't set --type.
            # TODO: Refine the logic to distinguish the case that --type not set and --type basic.
            module_meta={'name': name, 'job_type': None if job_type == 'basic' else job_type},
        )
        return entry_path

    @classmethod
    def load_from_project_file(cls, backup_folder):
        """Load module project from project file.

        :param backup_folder: backup folder
        :return: Returns a PipelineProject and list of module names failed to load.
        :rtype: (PipelineProject, List[str])
        """
        failed_modules = []
        handled_module_path = set()
        succeed_modules = []
        with open(cls.MODULE_PROJECT_FILE_NAME) as file:
            try:
                data = yaml.safe_load(file)
                # Avoid None is loaded from file
                if not isinstance(data, dict):
                    raise RuntimeError(f'Data {data} in {cls.MODULE_PROJECT_FILE_NAME} could not be loaded as a dict.')
            except Exception as e:
                logger.warning('Failed to load {}: {}\n'
                               'Error message: {}'.format(cls.MODULE_PROJECT_FILE_NAME, file.read(), str(e)))
                data = {}
            modules = data.get(cls.MODULES_KEY, [])
            for module_config in modules:
                # load from module_config
                try:
                    module_path, module_resource = _ModuleResource.load_from_config(module_config, backup_folder)
                except BaseException as e:
                    logger.warning('\t\tSkipped illegal config: {} due to error: {}'.format(module_config, e))
                    continue
                if module_resource is not None:
                    module_entry_path = module_resource.module_object.module_entry_path
                    if module_entry_path in handled_module_path:
                        logger.warning('Skipped duplicated module: {}'.format(module_path))
                    else:
                        handled_module_path.add(module_entry_path)
                        logger.info('\t\tLoaded: \t {}.'.format(module_entry_path))
                        succeed_modules.append(module_resource)
                else:
                    logger.warning('\t\tFailed to load {}.'.format(module_path))
                    failed_modules.append(module_path)

        pipeline_project = PipelineProject(succeed_modules)
        return pipeline_project, failed_modules

    @classmethod
    def load_from_py_file_list(cls, py_files: List, backup_folder):
        """Load module project from list of py files.

        :param py_files: python files
        :param backup_folder: backup folder
        :return: Returns a PipelineProject and list of module names failed to load.
        :rtype: (PipelineProject, List[str])
        """
        failed_modules = []
        modules = []
        for file in py_files:
            try:
                module_resource = _ModuleResource.from_dsl_module_file(file, backup_folder,
                                                                       raise_import_error=True)
                if module_resource is not None:
                    modules.append(module_resource)
                    logger.info('\t\tLoaded: \t {}.'.format(file))
            except ImportError:
                logger.warning('\t\tFailed to load {}.'.format(file))
                failed_modules.append(file)
        pipeline_project = PipelineProject(modules)
        return pipeline_project, failed_modules

    @classmethod
    def load_from_py_file(cls, py_file, backup_folder):
        """Load module project from file.

        :param py_file: python file
        :param backup_folder: backup folder
        :return: Returns a PipelineProject and list of module names failed to load.
        :rtype: (PipelineProject, List[str])
        """
        py_files = [py_file]
        return cls.load_from_py_file_list(py_files, backup_folder)

    @classmethod
    def load_from_folder(cls, folder, backup_folder):
        """Load module project from folder.

        :param backup_folder: backup folder
        :return: Returns a PipelineProject and list of module names failed to load.
        :rtype: (PipelineProject, List[str])
        """
        py_files = _find_py_files_in_target(folder)
        return cls.load_from_py_file_list(py_files, backup_folder)

    @classmethod
    def load(cls, target: Path, backup_folder):
        """Load a module project.

        :param target: file or directory
        :param backup_folder: backup folder
        :return: Returns a PipelineProject and list of module names failed to load.
        :rtype: (PipelineProject, List[str])
        """
        if target.is_file():
            if is_py_file(str(target)):
                logger.info('Loading dsl.modules from file...')
                return cls.load_from_py_file(target, backup_folder)
            else:
                raise ValueError('Target %s not valid.' % str(target))
        elif target.is_dir():
            if (target / cls.MODULE_PROJECT_FILE_NAME).exists():
                logger.info('Loading dsl.modules from {}...'.format(cls.MODULE_PROJECT_FILE_NAME))
                with _change_working_dir(target):
                    return cls.load_from_project_file(backup_folder)
            else:
                logger.info('Loading dsl.modules from folder...')
                return cls.load_from_folder(target, backup_folder)
        else:
            raise ValueError('Target %s not valid.' % str(target))

    @staticmethod
    def merge(first: 'PipelineProject', second: 'PipelineProject'):
        """Merge 2 module project, update first according to second."""
        second_modules_files = set([module.module_object.module_entry_path for module in second.modules])
        merged = []
        # add modules just in first
        for module in first.modules:
            if module.module_object.module_entry_path not in second_modules_files:
                merged.append(module)
        merged += second.modules
        first.modules = merged


def _entry(argv):
    """CLI tool for module creating."""
    parser = argparse.ArgumentParser(
        prog="python -m azureml.pipeline.wrapper.dsl.pipeline_project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""A CLI tool for module project creating"""
    )

    subparsers = parser.add_subparsers()

    # create module folder parser
    create_parser = subparsers.add_parser(
        'init',
        description='Add a dsl.module and resources into a pipeline project.'
                    'Specify name and type param to add a start up dsl.module and resources.'
                    'Specify function to build dsl.module and resources according to the function.'
                    'Specify file or module to build resources for existing dsl.module file.'
    )
    create_parser.add_argument(
        "--source", type=str,
        help="Source for specific mode, could be pacakge.function or path/to/python_file.py."
    )
    create_parser.add_argument(
        '--name', type=str,
        help="Name of the module."
    )
    create_parser.add_argument(
        '--type', type=str, default='basic', choices=['basic', 'mpi', 'parallel'],
        help="Job type of the module. Could be basic and mpi. "
             "Defaults to basic if not specified, which refers to run job on a single compute node."
    )
    create_parser.add_argument(
        '--source_dir', type=str,
        help="Source directory to init environment, all resources will be generated there, "
             "will be os.cwd() if not set."
    )
    create_parser.add_argument(
        '--inputs', type=str, default=[], nargs='+',
        help="Input ports of the module.",
    )
    create_parser.add_argument(
        '--outputs', type=str, default=[], nargs='+',
        help="Output ports of the module.",
    )
    create_parser.set_defaults(func=PipelineProject.init)

    # build dsl.module into specs parser
    build_parser = subparsers.add_parser(
        'build',
        description='A CLI tool to build dsl.module into module specs in folder.'
    )
    build_parser.add_argument(
        '--target', type=str,
        help="Target module project or module file. Will use current working directory if not specified."
    )
    build_parser.add_argument(
        '--source_dir', type=str,
        help="Source directory to build spec, will be os.cwd() if not set."
    )
    build_parser.set_defaults(func=PipelineProject.build)

    # This command is used for the case that one has a valid python entry with argparse,
    # but doesn't want to rely on dsl.module to register as a module.
    # Internal users may use this feature to build built-in modules.
    build_argparse_parser = subparsers.add_parser(
        'build-argparse',
        description="A CLI tool to build an entry with argparse into a module spec file."
    )
    build_argparse_parser.add_argument(
        '--target', type=str,
        help="Target entry file 'xx/yy.py' or target entry module 'xx.yy', file must be relative path.",
    )
    build_argparse_parser.add_argument(
        '--spec-file', type=str, default='spec.yaml',
        help="Module spec file name, the default name is 'spec.yaml', must be relative path",
    )
    build_argparse_parser.add_argument(
        '--source-dir', type=str, default='.',
        help="Source directory of the target file and spec file, the default path is '.'.",
    )
    build_argparse_parser.add_argument(
        '--inputs', type=str, default=[], nargs='+',
        help="Input ports of the module.",
    )
    build_argparse_parser.add_argument(
        '--outputs', type=str, default=[], nargs='+',
        help="Output ports of the module.",
    )
    build_argparse_parser.add_argument(
        '--force', action='store_true',
        help="Force generate spec file if exists, otherwise raises.",
    )
    build_argparse_parser.set_defaults(func=gen_module_by_argparse)

    args, rest_args = parser.parse_known_args(argv)
    if args.func == PipelineProject.init:
        PipelineProject.init(
            source=args.source, name=args.name, job_type=args.type,
            source_dir=args.source_dir,
            inputs=args.inputs, outputs=args.outputs,
        )
    elif args.func == PipelineProject.build:
        PipelineProject.build(target=args.target, source_dir=args.source_dir)
    elif args.func == gen_module_by_argparse:
        gen_module_by_argparse(
            entry=args.target, spec_file=args.spec_file, working_dir=args.source_dir,
            force=args.force, inputs=args.inputs, outputs=args.outputs,
        )


def main():
    """Use as a CLI entry function to use ModuleProject."""
    _entry(sys.argv[1:])


if __name__ == '__main__':
    main()
