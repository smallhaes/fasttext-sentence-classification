# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""A wrapper to analyze function annotations, generate module specs, and run module in command line."""
import copy
import types
import inspect
import argparse
import re
import sys
import functools
import contextlib
import multiprocessing
import importlib
from multiprocessing.pool import ThreadPool
from typing import List
from abc import ABCMeta
from collections import OrderedDict
from enum import EnumMeta
from pathlib import Path
from io import StringIO

from azureml._project.project_manager import _get_tagged_image

from azureml.pipeline.wrapper.dsl._utils import logger, _import_module_with_working_dir, _relative_to
from azureml.pipeline.wrapper.dsl._module_spec import BaseModuleSpec, ParallelRunModuleSpec,\
    InputPort, OutputPort, Param, _BaseParam, SPEC_EXT


OPENMPI_CPU_IMAGE = _get_tagged_image("mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu16.04")


# Need to consider where to put these logic.


def _to_camel_case(s):
    """Get the name in spec according to variable name."""
    s = s.replace('_', ' ')
    return s[0].upper() + s[1:]


def _sanitize_python_variable_name(s):
    """Get the variable name according to the name in spec, it should be the inverse of _to_camel_case."""
    return s.lower().replace(' ', '_')


def module(
    name=None, version=None, namespace=None,
    job_type=None, description=None,
    is_deterministic=None,
    tags=None, contact=None, help_document=None,
    os=None,
    base_image=None, conda_dependencies=None,
    parallel_inputs=None,
):
    """Return a decorator which is used to declare a module with @dsl.module.

    A module is a reusable unit in an Azure Machine Learning workspace.
    With the decorator @dsl.module, a function could be registered as a module in the workspace.
    Then the module could be used in an Azure Machine Learning pipeline.
    The parameters of the decorator are the properties of the module spec, see https://aka.ms/azureml-module-specs.

    .. remarks::

        The following example shows how to use @dsl.module to declare a simple module.

        .. code-block:: python

            @dsl.module
            def your_module_function(output: OutputDirectory(), input: InputDirectory(), param='str_param'):
                pass

        The following example shows how to declare a module with detailed meta data.

        .. code-block:: python

            @dsl.module(name=name, version=version, namespace=namespace, job_type=job_type, description=description)
            def your_module_function(output: OutputDirectory(), input: InputDirectory(), param='str_param'):
                pass

        A executable module should be in an entry file and could handle command line arguments.
        The following code is a full example of entry.py which could be registered.

        .. code-block:: python

            import sys
            from azureml.pipeline.wrapper import dsl
            from azureml.pipeline.wrapper.dsl.module import ModuleExecutor

            @dsl.module
            def your_module_function(output: OutputDirectory(), input: InputDirectory(), param='str_param'):
                pass

            if __name__ == '__main__':
                ModuleExecutor(your_module_function).execute(sys.argv)

        With the entry.py file, we could build a module specification yaml file.
        For more details of the module spec, see https://aka.ms/azureml-module-specs
        With the yaml file, we could register the module to to workspace using az ml cli.
        See https://docs.microsoft.com/en-us/cli/azure/ext/azure-cli-ml/ml?view=azure-cli-latest.
        The command lines are as follows.
        az ml module build --target entry.py
        az ml module register --spec-file entry.spec.yaml


    :param name: The name of the module. If None is set, camel cased function name is used.
    :type name: str
    :param description: The description of the module. If None is set, the doc string is used.
    :type description: str
    :param version: Version of the module.
    :type version: str
    :param namespace: Namespace of the module.
    :type namespace: str
    :param job_type: Job type of the module.
    :type job_type: str
    :param is_deterministic: Specify whether the module will always generate the same result.
    :type is_deterministic: bool
    :param tags: Tags of the module.
    :type tags: builtin.list
    :param contact: Contact of the module.
    :type contact: str
    :param help_document: Help document of the module.
    :type help_document: str
    :param os: OS type of the module.
    :type os: str
    :param base_image: Base image of the module.
    :type base_image: str
    :param conda_dependencies: Dependencies of the module.
    :type conda_dependencies: str
    :param parallel_inputs: A list of :class:`azureml.pipeline.wrapper.dsl.module.InputDirectory` object.
                            The inputs indicate the data for a parallel run module,
                            these inputs will be converted to a file list to be processed.
    :type parallel_inputs: builtin.list

    :return: An injected function which could be passed to ModuleExecutor
    """
    if os and os.lower() not in {'windows', 'linux'}:
        raise ValueError("Keyword 'os' only support two values: 'windows', 'linux'.")

    # For mpi module, due to some bug, IntelMPI cannot be initialized in linux docker in Windows OS.
    # See https://community.intel.com/t5/Intel-oneAPI-HPC-Toolkit/Intel-MPI-segmentation-fault-bug/td-p/1154073
    # To enable Module.run test in Windows OS for mpi module, we use openmpi base image instead.
    if base_image is None and job_type and job_type.lower() == 'mpi':
        base_image = OPENMPI_CPU_IMAGE
    spec_args = {k: v for k, v in locals().items() if v is not None}
    wrap_callable = False
    if callable(name):
        wrap_callable = True
        spec_args = {}

    def wrapper(func):
        spec_args['annotations'] = {'codegenBy': 'dsl.module'}  # Indicate the module is generated by dsl.module
        spec_args['name'] = spec_args.get('name', _to_camel_case(func.__name__))
        spec_args['description'] = spec_args.get('description', func.__doc__)
        entry, source_dir = _infer_func_relative_path_with_source(func)
        spec_args['command'] = ['python', entry]
        spec_args['source_directory'] = source_dir
        # Initialize a ModuleExecutor to make sure it works and use it to update the module function.
        executor = ModuleExecutor(func, copy.copy(spec_args))
        executor._update_func(func)
        return func

    return wrapper(name) if wrap_callable else wrapper


def _infer_func_relative_path_with_source(func):
    """Infer the relative python entry file which could be correctly run in AzureML."""
    # If the function is imported from xx.yy, we simply use xx/yy.py as entry, then infer source dir.
    func_entry_path = Path(inspect.getfile(func)).resolve().absolute()
    if func.__module__ != '__main__' and not func.__module__.startswith('<'):
        entry = func.__module__.replace('.', '/') + '.py'
        source_dir = func_entry_path
        for _ in func.__module__.split('.'):
            source_dir = source_dir.parent
        return entry, source_dir
    # Otherwise it is in the main file.
    working_dir = Path('.').resolve().absolute()
    relative_path = _relative_to(func_entry_path, working_dir)
    # If the file path is under the working directory,
    # the source directory should be working dir, the path should be the relative path from working dir.
    if relative_path:
        return relative_path.as_posix(), working_dir
    # Otherwise we simply use the filename as the function relative path, its parent as source.
    else:
        return func_entry_path.name, func_entry_path.parent


class RequiredParamParsingError(ValueError):
    """This error indicates that a parameter is required but not exists in the command line."""

    def __init__(self, name, arg_string):
        """Init the error with the parameter name and its arg string."""
        msg = "'%s' cannot be None since it is not optional. " % name + \
              "Please make sure command option '%s' exists." % arg_string
        super().__init__(msg)


class _ModuleBaseParam(_BaseParam):
    """This class defines some common operation for ModuleInputPort/ModuleOutputPort/ModuleParam."""

    @property
    def arg_string(self):
        """Compute the cli option str according to its name, used in argparser."""
        return '--' + self.to_var_name()

    def to_cli_option_str(self, style=None):
        """Return the cli option str with style, by default return underscore style --a_b."""
        return self.arg_string.replace('_', '-') if style == 'hyphen' else self.arg_string

    def to_var_name(self):
        """Get the variable name in python."""
        return _sanitize_python_variable_name(self.name)

    def update_name(self, name):
        """Update the name of the port/param.

        Initially the names of inputs should be None, then we use variable names of python function to update it.
        """
        if self._name is not None:
            raise AttributeError("Cannot set name to %s since it is not None, the value is %s." % (name, self._name))
        self._name = name

    def add_to_arg_parser(self, parser: argparse.ArgumentParser, default=None):
        """Add this parameter to ArgumentParser, both command line styles are added."""
        cli_str_underscore = self.to_cli_option_str(style='underscore')
        cli_str_hyphen = self.to_cli_option_str(style='hyphen')
        if default is not None:
            return parser.add_argument(cli_str_underscore, cli_str_hyphen, default=default)
        else:
            return parser.add_argument(cli_str_underscore, cli_str_hyphen,)

    def set_optional(self):
        """Set the parameter as an optional parameter."""
        self._optional = True

    @classmethod
    def register_data_type(cls, data_type: type):
        """Register the data type to the corresponding parameter/port."""
        if not isinstance(data_type, type):
            raise TypeError("Only python type is allowed to register, got %s." % data_type)
        cls.DATA_TYPE_MAPPING[data_type] = cls


class _DataTypeRegistrationMeta(ABCMeta):
    """This meta class is used to register data type mapping for ports/parameters.

    With this metaclass, a simple annotation str could be converted to StringParameter by declaring `DATA_TYPE`.
    """

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)
        data_type = getattr(cls, 'DATA_TYPE', None)
        if data_type is not None:
            try:
                cls()
            except TypeError:
                raise ValueError("To register a data type, the class must be able to initialized with %s()" % name)
            cls.register_data_type(data_type)
        return cls


class _ModuleParam(Param, _ModuleBaseParam, metaclass=_DataTypeRegistrationMeta):
    """This is the base class of module parameters.

    The properties including name/type/default/options/optional/min/max will be dumped in module spec.
    When invoking a module, param.parse_and_validate(str_val) is called to parse the command line value.
    """

    DATA_TYPE_MAPPING = {}

    def __init__(self, name, type,
                 description=None, default=None, options=None, optional=False, min=None, max=None,
                 ):
        super().__init__(name, type, description, default, options, optional, min, max)
        self._allowed_types = ()
        data_type = getattr(self, 'DATA_TYPE', None)
        # TODO: Maybe a parameter could have several allowed types? For example, json -> List/Dict?
        if data_type:
            self._allowed_types = (data_type,)
        self.update_default(default)

    def update_default(self, default_value):
        """Update default is used when the annotation has default values.

        Here we need to make sure the type of default value is allowed.
        """
        if default_value is not None and not isinstance(default_value, self._allowed_types):
            try:
                default_value = self.parse(default_value)
            except Exception as e:
                if self.name is None:
                    msg = "Default value of %s cannot be parsed, got '%s', type = %s." % (
                        type(self).__name__, default_value, type(default_value)
                    )
                else:
                    msg = "Default value of %s '%s' cannot be parsed, got '%s', type = %s." % (
                        type(self).__name__, self.name, default_value, type(default_value)
                    )
                raise TypeError(msg) from e
        self._default = default_value

    def parse(self, str_val: str):
        """Parse str value passed from command line."""
        return str_val

    def validate_or_throw(self, value):
        """Validate input parameter value, throw exception if not as required.

        It will throw exception if validate failed, otherwise do nothing.
        """
        if not self.optional and value is None:
            raise ValueError("Parameter %s cannot be None since it is not optional." % self.name)
        if self._allowed_types and value is not None:
            if not isinstance(value, self._allowed_types):
                raise TypeError(
                    "Unexpected data type for parameter '%s'. Expected %s but got %s." % (
                        self.name, self._allowed_types, type(value)
                    )
                )

    def parse_and_validate(self, value):
        """Parse the value and validate it."""
        value = self.parse(value) if isinstance(value, str) else value
        self.validate_or_throw(value)
        return value

    def add_to_arg_parser(self, parser: argparse.ArgumentParser, default=None):
        """Add this parameter to ArgumentParser with its default value."""
        default = default or self.default
        super().add_to_arg_parser(parser, default)


class _ModuleInputPort(InputPort, _ModuleBaseParam):
    """This is the base class of module input ports.

    The properties including type/description/optional will be dumped in module spec.
    """

    def __init__(self, type, description=None, name=None, optional=None):
        """Initialize an input port."""
        super().__init__(name=name, type=type, description=description, optional=optional)

    def load(self, str_val: str):
        """Load the data from an input_path with type str."""
        return str_val


class InputDirectory(_ModuleInputPort):
    """InputDirectory indicates an input which is a directory."""

    def __init__(self, type='AnyDirectory', description=None, name=None, optional=None):
        """Initialize an output directory port, declare type to use your custmized port type."""
        super().__init__(type=type, description=description, name=name, optional=optional)


class InputFile(_ModuleInputPort):
    """InputFile indicates an input which is a file."""

    def __init__(self, type='AnyFile', description=None, name=None, optional=None):
        """Initialize an input file port Declare type to use your custmized port type."""
        super().__init__(type=type, description=description, name=name, optional=optional)


class _InputFileList:

    def __init__(self, inputs: List[_ModuleInputPort]):
        self.validate_inputs(inputs)
        self._inputs = inputs

    @classmethod
    def validate_inputs(cls, inputs):
        for i, port in enumerate(inputs):
            if not isinstance(port, (InputFile, InputDirectory)):
                raise TypeError("You could only use InputFile/InputDirectory in an input list, got '%s'." % type(port))
            if port.name is None:
                raise ValueError("You must specify the name of the %dth input." % i)
        if all(port.optional for port in inputs):
            raise ValueError("You must specify at least 1 required port in the input list, got 0.")

    def add_to_arg_parser(self, parser: argparse.ArgumentParser):
        for port in self._inputs:
            port.add_to_arg_parser(parser)

    def load_from_args(self, args):
        """Load the input files from parsed args from ArgumentParser."""
        files = []
        for port in self._inputs:
            str_val = getattr(args, port.to_var_name(), None)
            if str_val is None:
                if not port.optional:
                    raise RequiredParamParsingError(name=port.name, arg_string=port.arg_string)
                continue
            files += [str(f) for f in Path(str_val).glob('**/*') if f.is_file()]
        return files

    def load_from_argv(self, argv=None):
        if argv is None:
            argv = sys.argv
        parser = argparse.ArgumentParser()
        self.add_to_arg_parser(parser)
        args, _ = parser.parse_known_args(argv)
        return self.load_from_args(args)

    @property
    def inputs(self):
        return self._inputs


class _ModuleOutputPort(OutputPort, _ModuleBaseParam):
    """This is the base class of module output ports.

    The properties including type/description will be dumped in module spec.
    """

    def __init__(self, type, description=None):
        super().__init__(name=None, type=type, description=description)

    def set_optional(self):
        """Set output port as optional always fail."""
        pass


class OutputDirectory(_ModuleOutputPort):
    """OutputDirectory indicates an output which is a directory."""

    def __init__(self, type='AnyDirectory', description=None):
        """Initialize an output directory port, declare type to use your custmized port type."""
        super().__init__(type=type, description=description)


class OutputFile(_ModuleOutputPort):
    """OutputFile indicates an output which is a file."""

    def __init__(self, type='AnyFile', description=None):
        """Initialize an output file port Declare type to use your custmized port type."""
        super().__init__(type=type, description=description)


class StringParameter(_ModuleParam):
    """String parameter passed the parameter string with its raw value."""

    DATA_TYPE = str

    def __init__(
            self,
            description=None,
            optional=False,
            default=None,
    ):
        """Initialize a bool parameter."""
        super().__init__(
            name=None,
            description=description,
            optional=optional,
            default=default,
            type='String',
        )


class EnumParameter(_ModuleParam):
    """Enum parameter parse the value according to its enum values."""

    def __init__(
            self,
            enum: EnumMeta = None,
            description=None,
            optional=False,
            default=None,
    ):
        """Initialize an enum parameter, the options of an enum parameter are the enum values."""
        if not isinstance(enum, EnumMeta):
            raise ValueError("enum must be a subclass of Enum.")
        if len(list(enum)) <= 0:
            raise ValueError("enum must have enum values.")
        self._enum = enum
        self._option2enum = {str(option.value): option for option in enum}
        self._val2enum = {option.value: option for option in enum}
        super().__init__(
            name=None,
            optional=optional,
            description=description,
            default=default,
            type='Enum',
            options=[str(option.value) for option in enum],
        )

    def parse(self, str_val: str):
        """Parse the enum value from a string value."""
        if str_val not in self._option2enum and str_val not in self._val2enum:
            raise ValueError("Not a valid enum value: '%s', valid values: %s" % (str_val, ', '.join(self.options)))
        return self._option2enum.get(str_val) or self._val2enum.get(str_val)

    def update_default(self, default_value):
        """Enum parameter support updating values with a string value."""
        if default_value in self._val2enum:
            default_value = self._val2enum[default_value]
        if isinstance(default_value, self._enum):
            default_value = default_value.value
        if default_value is not None and default_value not in self._option2enum:
            raise ValueError(
                "Not a valid enum value: '%s', valid values: %s" % (default_value, ', '.join(self.options))
            )
        self._default = default_value


class _NumericParameter(_ModuleParam):
    """Numeric Parameter is an intermediate type which is used to validate the value according to min/max."""

    def validate_or_throw(self, val):
        super().validate_or_throw(val)
        if self._min is not None and val < self._min:
            raise ValueError("Parameter '%s' should not be less than %s." % (self.name, self._min))
        if self._max is not None and val > self._max:
            raise ValueError("Parameter '%s' should not be greater than %s." % (self.name, self._max))


class IntParameter(_NumericParameter):
    """Int Parameter parse the value to a int value."""

    DATA_TYPE = int

    def __init__(
            self,
            min=None,
            max=None,
            description=None,
            optional=False,
            default=None,
    ):
        """Initialize an integer parameter."""
        super().__init__(
            name=None,
            optional=optional,
            description=description,
            default=default,
            min=min,
            max=max,
            type='Integer',
        )

    def parse(self, val):
        """Parse the integer value from a string value."""
        return int(val)


class FloatParameter(_NumericParameter):
    """Float Parameter parse the value to a float value."""

    DATA_TYPE = float

    def __init__(
            self,
            min=None,
            max=None,
            description=None,
            optional=False,
            default=None,
    ):
        """Initialize a float parameter."""
        super().__init__(
            name=None,
            optional=optional,
            description=description,
            default=default,
            min=min,
            max=max,
            type='Float',
        )

    def parse(self, val):
        """Parse the float value from a string value."""
        return float(val)


class BoolParameter(_ModuleParam):
    """Bool Parameter parse the value to a bool value."""

    DATA_TYPE = bool

    def __init__(
            self,
            description=None,
            default=False,
    ):
        """Initialize a bool parameter."""
        super().__init__(
            name=None,
            optional=True,
            description=description,
            default=default,
            type='Boolean',
        )

    def parse(self, val):
        """Parse the bool value from a string value."""
        if val.lower() not in {'true', 'false'}:
            raise ValueError("Bool parameter '%s' only accept True/False, got %s." % (self.name, val))
        return True if val.lower() == 'true' else False


class TooManyDSLModulesError(Exception):
    """Exception when multiple dsl.modules are found in single module entry."""

    def __init__(self, count, file):
        """Error message inits here."""
        super().__init__("Only one dsl.module is allowed per file, {} found in {}".format(count, file))


class ModuleExecutor:
    """An executor to analyze the spec args of a function and convert it to a runnable module in AzureML."""

    INJECTED_FIELD = '_spec_args'  # The injected field is used to get the module spec args of the function.

    def __init__(self, func: types.FunctionType, spec_args=None):
        """Initialize a ModuleExecutor with a function to enable calling the function with command line args.

        :param func: A function wrapped by dsl.module.
        :type func: types.FunctionType
        """
        if not isinstance(func, types.FunctionType):
            raise TypeError("Only function type is allowed to initialize ModuleExecutor.")
        if spec_args is None:
            spec_args = getattr(func, self.INJECTED_FIELD, None)
            if spec_args is None:
                raise TypeError("You must wrap the function with @dsl.module() before using it.")
        self._raw_spec_args = copy.copy(spec_args)
        self._name = spec_args['name']
        self._job_type = spec_args.get('job_type', 'basic')
        executor_cls = self._get_executor_by_job_type(self.job_type)
        if executor_cls is None:
            raise ValueError("Unrecognized job_type '%s' of dsl.module '%s'." % (self.job_type, func.__name__))
        self._executor = executor_cls(func, spec_args=spec_args)
        self._func = func

    @property
    def name(self):
        """Return the name of the module."""
        return self._name

    @property
    def job_type(self):
        """Return the job type of the module."""
        return self._job_type

    @property
    def spec(self):
        """Return the module spec instance of the module, initialized by the function annotations and the meta data."""
        return self._executor.spec

    @property
    def spec_dict(self):
        """Return the module spec data as a python dict."""
        return self._executor.spec.spec_dict

    def to_spec_yaml(self, folder=None, spec_file=None):
        """Generate spec dict object, and dump it as a yaml spec file."""
        pyfile = Path(inspect.getfile(self._func))
        if folder is None:
            # If the folder is not provided, we generate the spec file in the same folder of the function file.
            folder = pyfile.parent
        if spec_file is None:
            # If the spec file name is not provided, get the name from the file name.
            spec_file = pyfile.with_suffix(SPEC_EXT).name
        self.spec.dump_module_spec_to_folder(folder, spec_file)
        return Path(folder) / spec_file

    def get_interface(self):
        """Return the interface of this module.

        :return: A dictionary including the definition of inputs/outputs/params.
        """
        return self._executor.get_interface()

    def execute(self, argv):
        """Execute the module with command line arguments."""
        return self._executor.execute(argv)

    def __call__(self, *args, **kwargs):
        """Directly calling a module executor equals to calling the underlying function directly."""
        return self._func(*args, **kwargs)

    @classmethod
    def collect_module_from_file(cls, py_file, working_dir=None, force_reload=False):
        """Collect single dsl module in a file and return the executors of the modules."""
        py_file = Path(py_file).absolute()
        if py_file.suffix != '.py':
            raise ValueError("%s is not a valid py file." % py_file)
        if working_dir is None:
            working_dir = py_file.parent
        working_dir = Path(working_dir).absolute()

        module_path = py_file.relative_to(working_dir).as_posix().split('.')[0].replace('/', '.')

        return cls.collect_module_from_py_module(module_path, working_dir=working_dir, force_reload=force_reload)

    @classmethod
    def collect_module_from_py_module(cls, py_module, working_dir, force_reload=False):
        """Collect single dsl module in a py module and return the executors of the modules."""
        modules = [module for module in cls.collect_modules_from_py_module(py_module, working_dir, force_reload)]

        def defined_in_current_file(module):
            entry_file = inspect.getfile(module._func)
            module_path = py_module.replace('.', '/') + '.py'
            return Path(entry_file).resolve().absolute() == (Path(working_dir) / module_path).resolve().absolute()

        modules = [module for module in modules if defined_in_current_file(module)]
        if len(modules) == 0:
            return None
        module = modules[0]
        entry_file = inspect.getfile(module._func)
        if len(modules) > 1:
            raise TooManyDSLModulesError(len(modules), entry_file)
        module.check_entry_valid(entry_file)
        return module

    @classmethod
    def collect_modules_from_py_module(cls, py_module, working_dir=None, force_reload=False):
        """Collect all modules in a python module and return the executors of the modules."""
        if isinstance(py_module, str):
            try:
                py_module = _import_module_with_working_dir(py_module, working_dir, force_reload)
            except Exception as e:
                raise ImportError("""Error occurs when import module '%s': %s.""" % (py_module, e)) from e
        for _, obj in inspect.getmembers(py_module):
            if cls.look_like_module(obj):
                module = ModuleExecutor(obj)
                module.check_py_module_valid(py_module)
                yield module

    @classmethod
    def look_like_module(cls, f):
        """Return True if f looks like a module."""
        if not isinstance(f, types.FunctionType):
            return False
        if not hasattr(f, cls.INJECTED_FIELD):
            return False
        return True

    @classmethod
    def _get_executor_by_job_type(cls, job_type):
        executors = [_BasicModuleExecutor, _ParallelModuleExecutor]
        for executor in executors:
            if executor.is_valid_job_type(job_type):
                return executor
        return None

    def check_entry_valid(self, entry_file):
        """Check whether the entry file is valid to make sure it could be run in AzureML."""
        return self._executor.check_entry_valid(entry_file)

    def check_py_module_valid(self, py_module):
        """Check whether the entry py module is valid to make sure it could be run in AzureML."""
        return self._executor.check_py_module_valid(py_module)

    def _update_func(self, func: types.FunctionType):
        # Set the injected field so the function could be used to initializing with `ModuleExecutor(func)`
        setattr(func, self.INJECTED_FIELD, self._raw_spec_args)
        if hasattr(self._executor, '_update_func'):
            self._executor._update_func(func)


class _BasicModuleExecutor:
    SPEC_CLASS = BaseModuleSpec  # This class is used to initialize a module spec instance.
    SPECIAL_FUNC_CHECKERS = {
        'Coroutine': inspect.iscoroutinefunction,
        'Async generator': inspect.isasyncgenfunction,
        'Generator': inspect.isgeneratorfunction,
    }
    VALID_SPECIAL_FUNCS = set()

    def __init__(self, func: types.FunctionType, spec_args=None):
        """Initialize a ModuleExecutor with a function."""
        if spec_args is None:
            spec_args = getattr(func, ModuleExecutor.INJECTED_FIELD)
        self._spec_args = copy.deepcopy(spec_args)
        self._assert_job_type(self.job_type)
        self._assert_valid_func(func)
        self._func = func
        self._arg_mapping = self._analyze_annotations(func)
        self._parallel_inputs = None
        if 'parallel_inputs' in spec_args:
            self._parallel_inputs = _InputFileList(self._spec_args.pop('parallel_inputs'))

    @property
    def job_type(self):
        return self._spec_args.get('job_type', 'basic')

    @property
    def spec(self):
        """Return the module spec instance of the module, initialized by the function annotations and the meta data."""
        io_properties = self._generate_spec_io_properties(self._arg_mapping, self._parallel_inputs)
        return self.SPEC_CLASS(**self._spec_args, **io_properties)

    def get_interface(self):
        """Return the interface of this module.

        :return: A dictionary including the definition of inputs/outputs/params.
        """
        properties = self._generate_spec_io_properties(self._arg_mapping, self._parallel_inputs)
        properties.pop('args')
        return properties

    def execute(self, argv):
        """Execute the module with command line arguments."""
        args = self._parse(argv)
        run = self._func(**args)
        if self._parallel_inputs is not None:
            run(self._parallel_inputs.load_from_argv(argv))

    def __call__(self, *args, **kwargs):
        """Directly calling a module executor equals to calling the underlying function directly."""
        return self._func(*args, **kwargs)

    @classmethod
    def is_valid_job_type(cls, job_type):
        return job_type in {None, 'mpi', 'basic'}

    @classmethod
    def _assert_job_type(cls, job_type):
        if not cls.is_valid_job_type(job_type):
            raise ValueError("Job type '%s' is invalid for '%s'" % (job_type, cls.__name__))

    def _assert_valid_func(self, func):
        """Check whether the function is valid, if it is not valid, raise."""
        for k, checker in self.SPECIAL_FUNC_CHECKERS.items():
            if k not in self.VALID_SPECIAL_FUNCS:
                if checker(func):
                    raise NotImplementedError("%s function is not supported for %s module now." % (k, self.job_type))

    def check_entry_valid(self, entry_file):
        """Check whether the entry file call .execute(sys.argv) to make sure it could be run in AzureML."""
        # Now we simply use string search, will be refined in the future.
        main_code = """if __name__ == '__main__':\n    ModuleExecutor(%s).execute(sys.argv)""" % self._func.__name__
        with open(entry_file) as fin:
            if main_code not in fin.read():
                msg = "The following code doesn't exist in the entry file, it may not run correctly.\n%s" % main_code
                logger.warning(msg)

    def check_py_module_valid(self, py_module):
        pass

    @classmethod
    def _parse_with_mapping(cls, argv, arg_mapping):
        """Use the parameters info in arg_mapping to parse commandline params.

        :param argv: Command line arguments like ['--param-name', 'param-value']
        :param arg_mapping: A dict contains the mapping from param key 'param_name' to _ModuleBaseParam
        :return: params: The parsed params used for calling the user function.
        """
        parser = argparse.ArgumentParser()
        for param in arg_mapping.values():
            param.add_to_arg_parser(parser)
        args, _ = parser.parse_known_args(argv)

        # Convert the string values to real params of the function.
        params = {}
        for name, param in arg_mapping.items():
            val = getattr(args, param.to_var_name())
            if val is None:
                if isinstance(param, _ModuleOutputPort) or not param.optional:
                    raise RequiredParamParsingError(name=param.name, arg_string=param.arg_string)
                continue
            # If it is a parameter, we help the user to parse the parameter,
            # if it is an input port, we use load to get the param value of the port,
            # otherwise we just pass the raw value as the param value.
            param_value = val
            if isinstance(param, _ModuleParam):
                param_value = param.parse_and_validate(val)
            elif isinstance(param, _ModuleInputPort):
                param_value = param.load(val)
            params[name] = param_value
            # For OutputDirectory, we will create a folder for it.
            if isinstance(param, OutputDirectory) and not Path(val).exists():
                Path(val).mkdir(parents=True, exist_ok=True)
        return params

    def _parse(self, argv):
        return self._parse_with_mapping(argv, self._arg_mapping)

    @classmethod
    def _generate_spec_outputs(cls, arg_mapping):
        """Generate output ports of a module, from the return annotation and the arg annotations.

        The outputs including the return values and the special PathOutputPort in args.
        """
        return [val for val in arg_mapping.values() if isinstance(val, _ModuleOutputPort)]

    @classmethod
    def _generate_spec_inputs(cls, arg_mapping, parallel_inputs: _InputFileList = None):
        """Generate input ports of the module according to the analyzed argument mapping."""
        input_ports = [val for val in arg_mapping.values() if isinstance(val, _ModuleInputPort)]
        if parallel_inputs:
            input_ports = [port for port in parallel_inputs.inputs] + input_ports
        return input_ports

    @classmethod
    def _generate_spec_params(cls, arg_mapping):
        """Generate parameters of the module according to the analyzed argument mapping."""
        return [val for val in arg_mapping.values() if isinstance(val, _ModuleParam)]

    @classmethod
    def _generate_spec_io_properties(cls, arg_mapping, parallel_inputs=None):
        """Generate the required properties for a module spec according to the annotation of a function."""
        inputs = cls._generate_spec_inputs(arg_mapping, parallel_inputs)
        outputs = cls._generate_spec_outputs(arg_mapping)
        params = cls._generate_spec_params(arg_mapping)
        args = []
        for val in inputs + params + outputs:
            if isinstance(val, (_ModuleInputPort, _ModuleParam)) and val.optional:
                args.append(val.arg_group())
            else:
                args += val.arg_group()
        return {'inputs': inputs, 'outputs': outputs, 'params': params, 'args': args}

    @classmethod
    def _analyze_annotations(cls, func):
        """Analyze the annotation of the function to get the parameter mapping dict and the output port list.

        :param func:
        :return: (param_mapping, output_list)
            param_mapping: The mapping from function param names to input ports/module parameters;
            output_list: The output port list analyzed from return annotations.
        """
        mapping = OrderedDict()
        sig = inspect.signature(func)
        for param_name, param_attr in sig.parameters.items():
            annotation = cls._generate_parameter_annotation(param_attr)
            if annotation.name is None:
                annotation.update_name(_to_camel_case(param_name))
            annotation.arg_name = param_name
            mapping[param_name] = annotation

        return mapping

    @classmethod
    def _generate_parameter_annotation(cls, param_attr):
        """Generate an input port/parameter according to a param annotation of the function."""
        annotation = param_attr.annotation

        # If the user forgot to initialize an instance, help him to initalize.
        if isinstance(annotation, type) and issubclass(annotation, _ModuleBaseParam):
            annotation = annotation()

        # If the param doesn't have annotation, we get the annotation from the default value.
        # If the default value is None or no default value, it is treated as str.
        if annotation is param_attr.empty:
            default = param_attr.default
            annotation = str if default is None or default is param_attr.empty else type(param_attr.default)

        # An enum type will be converted to EnumParameter
        if isinstance(annotation, EnumMeta):
            annotation = EnumParameter(enum=annotation)

        # If the annotation is not one of _ModuleParam/ModuleInputPort/ModulePort,
        # we use DATA_TYPE_MAPPING to get the corresponding class according to the type of annotation.
        if not isinstance(annotation, (_ModuleParam, _ModuleInputPort, _ModuleOutputPort)):
            param_cls = _ModuleParam.DATA_TYPE_MAPPING.get(annotation)
            if param_cls is None:
                # If the type is still unrecognized, we treat is as string.
                param_cls = StringParameter
            annotation = param_cls()
        annotation = copy.copy(annotation)

        # If the default value of a parameter is set, set the port/param optional,
        # and set the default value of a parameter.
        # Note that output port cannot be optional.
        if not isinstance(param_attr, _ModuleOutputPort) and param_attr.default is not param_attr.empty:
            annotation.set_optional()
            # Only parameter support default value in yaml.
            if isinstance(annotation, _ModuleParam):
                annotation.update_default(param_attr.default)

        return annotation

    def _update_func(self, func):
        pass


class _ParallelModuleExecutor(_BasicModuleExecutor):
    """This executor handle parallel module specific operations to enable parallel module."""

    SPEC_CLASS = ParallelRunModuleSpec
    JOB_TYPE = 'parallel'
    FIELDS = {'init', 'run', 'shutdown'}
    CONFLICT_ERROR_TPL = "It is not allowed to declare {}() once a parallel module is defined."
    VALID_SPECIAL_FUNCS = {'Generator'}

    def __init__(self, func: callable, spec_args=None):
        """Initialize a ParallelModuleExecutor with a provided function."""
        super().__init__(func, spec_args)
        if not self._parallel_inputs:
            raise ValueError(
                "Parallel module should have at lease one parallel input, got 0.",
            )
        self._output_keys = [key for key, val in self._arg_mapping.items() if isinstance(val, OutputDirectory)]
        if len(self._output_keys) == 0:
            raise ValueError(
                "Parallel module should have at least one OutputDirectory, got %d." % len(self._output_keys)
            )
        self._args = {}
        self._spec_args.update({
            'input_data': [port.name for port in self._parallel_inputs.inputs],
            # We use the first output as the parallel output data.
            # This is only a workaround according to current parallel run design, picking any output port is OK.
            'output_data': self._arg_mapping[self._output_keys[0]].name,
        })
        command = self._spec_args.pop('command')
        self._spec_args['entry'] = command[-1]
        self._spec_args.pop('job_type')
        self._run_func = None
        self._generator = None

    def execute(self, argv, batch_size=4):
        """Execute the module using parallel run style. This is used for local debugging."""
        self.init_argv(argv)

        files = self._parallel_inputs.load_from_argv(argv)
        # Use multiprocessing to run batches.
        count = len(files)
        batches = (count + batch_size - 1) // batch_size
        nprocess = min(max(batches, 1), multiprocessing.cpu_count())
        logger.info("Run %d batches to process %d files." % (batches, count))
        batch_files = [files[i * batch_size: (i + 1) * batch_size] for i in range(batches)]
        with ThreadPool(nprocess) as pool:
            batch_results = pool.map(self.run, batch_files)
        results = []
        for result in batch_results:
            results += result
        shutdown_result = self.shutdown()
        return shutdown_result if shutdown_result is not None else results

    @staticmethod
    def _remove_ambiguous_option_in_argv(argv: list, parse_method):
        """Remove ambiguous options in argv for an argparser method.

        This is a workaround to solve the issue that parallel run will add some other command options
        which will cause the problem 'ambiguous option'.
        """
        pattern = re.compile(r"error: ambiguous option: (\S+) could match")
        while True:
            stderr = StringIO()
            with contextlib.redirect_stderr(stderr):
                try:
                    parse_method(argv)
                except SystemExit:
                    stderr_value = stderr.getvalue()
                    match = pattern.search(stderr_value)
                    if not match:
                        # If we cannot found such pattern, which means other problems is raised, we directly raise.
                        sys.stdout.write(stderr_value)
                        raise
                    # Remove the option_str and the value of it.
                    option_str = match.group(1)
                    logger.debug("Ambiguous option '%s' is found in argv, remove it." % option_str)
                    idx = argv.index(option_str)
                    argv = argv[:idx] + argv[idx + 2:]
                else:
                    # If no exception is raised, return the ready args.
                    return argv

    def init(self):
        """Init params except for the InputFiles with the sys args when initializing parallel module.

        This method will only be called once in one process.
        """
        return self.init_argv(sys.argv)

    def init_argv(self, argv=None):
        """Init params except for the InputFiles with argv."""
        if argv is None:
            argv = sys.argv
        logger.info("Initializing parallel module, argv = %s" % argv)
        mapping = copy.copy(self._arg_mapping)
        argv = self._remove_ambiguous_option_in_argv(
            argv, functools.partial(self._parse_with_mapping, arg_mapping=mapping),
        )
        args = self._parse_with_mapping(argv, mapping)
        logger.info("Parallel module initialized, args = %s" % args)
        ret = self._func(**args)
        # If the init function is a generator, the first yielded result is the run function.
        if isinstance(ret, types.GeneratorType):
            self._generator = ret
            ret = next(ret)

        # Make sure the return result is a callable.
        if callable(ret):
            self._run_func = ret
        else:
            raise TypeError("Return/Yield result of the function must be a callable, got '%s'." % (type(ret)))

        sig = inspect.signature(self._run_func)
        if len(sig.parameters) != 1:
            raise ValueError(
                "The method {}() returned by {}() has incorrect signature {}."
                " It should have exact one parameter.".format(ret.__name__, self._func.__name__, sig)
            )
        return self._run_func

    def run(self, files):
        results = self._run_func(files)
        if results is not None:
            return files
        return results

    def shutdown(self):
        if self._generator:
            # If the function is using yield, call next to run the codes after yield.
            while True:
                try:
                    next(self._generator)
                except StopIteration as e:
                    return e.value

    def check_entry_valid(self, entry_file):
        pass

    def check_py_module_valid(self, py_module):
        # For parallel module, the init/run/shutdown in py_module should be _ParallelModuleExecutor.init/run/shutdown
        for attr in self.FIELDS:
            func = getattr(py_module, attr)
            if not self.is_valid_init_run_shutdown(func, attr):
                raise AttributeError(self.CONFLICT_ERROR_TPL.format(attr))

    def _update_func(self, func: types.FunctionType):
        # For a parallel module, we should update init/run/shutdown for the script.
        # See "Write your inference script" in the following link.
        # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-run-step
        py_module = importlib.import_module(func.__module__)
        for attr in self.FIELDS:
            func = getattr(py_module, attr, None)
            # We don't allow other init/run/shutdown in the script.
            if func is not None and not self.is_valid_init_run_shutdown(func, attr):
                raise AttributeError(self.CONFLICT_ERROR_TPL.format(attr))
            setattr(py_module, attr, getattr(self, attr))

    @classmethod
    def is_valid_init_run_shutdown(cls, func, attr):
        return isinstance(func, types.MethodType) and func.__func__ == getattr(_ParallelModuleExecutor, attr)

    @classmethod
    def _generate_spec_io_properties(cls, arg_mapping, parallel_inputs=None):
        """Generate the required properties for a module spec according to the annotation of a function.

        For parallel module, we need to remove InputFiles and --output in args.
        """
        properties = super()._generate_spec_io_properties(arg_mapping, parallel_inputs)
        args_to_remove = []
        for k, v in arg_mapping.items():
            # InputFiles and the output named --output need to be removed in the arguments.
            # For InputFiles: the control script will handle it and pass the files to run();
            # For the output, the control script will add an arg item --output so we should not define it again.
            if v.to_cli_option_str() == '--output':
                args_to_remove.append(v)
        if parallel_inputs:
            args_to_remove += [port for port in parallel_inputs.inputs]
        args = properties['args']
        for arg in args_to_remove:
            if isinstance(arg, _ModuleInputPort) and arg.optional:
                args.remove(arg.arg_group())
            else:
                idx = args.index(arg.arg_string)
                args.remove(arg.arg_string)
                args.remove(args[idx])
        return properties

    @classmethod
    def is_valid_job_type(cls, job_type):
        return job_type == cls.JOB_TYPE
