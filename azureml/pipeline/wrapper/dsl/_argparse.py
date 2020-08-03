# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import sys
import runpy
from pathlib import Path
from azureml.pipeline.wrapper.dsl._module_generator import ModuleGenerator, is_py_file, ParamGenerator
from azureml.pipeline.wrapper.dsl.module import _to_camel_case
from azureml.pipeline.wrapper.dsl._module_spec import Param
from azureml.pipeline.wrapper.dsl._utils import logger, inject_sys_path


def gen_module_by_argparse(
    entry: str, target_file=None, spec_file=None,
    working_dir=None, force=False,
    inputs=None, outputs=None,
    module_meta=None,
):
    if working_dir is None:
        working_dir = '.'

    if not Path(working_dir).is_dir():
        raise ValueError("Working directory '%s' is not a valid directory." % working_dir)

    is_file = is_py_file(entry)
    if is_file:
        if Path(entry).is_absolute():
            raise ValueError("Absolute file path '%s' is not allowed." % entry)
        if not (Path(working_dir) / entry).is_file():
            raise FileNotFoundError("Entry file '%s' not found in working directory '%s'." % (entry, working_dir))
        entry = Path(entry).as_posix()

    if target_file:
        if Path(target_file).is_absolute():
            raise ValueError("Absolute target file path '%s' is not allowed." % target_file)
        if not target_file.endswith('.py'):
            raise ValueError("Target file must has extension '.py', got '%s'." % target_file)
        if not force and (Path(working_dir) / target_file).exists():
            raise FileExistsError("Target file '%s' already exists." % target_file)

    if spec_file:
        if Path(spec_file).is_absolute():
            raise ValueError("Absolute module spec file path '%s' is not allowed." % spec_file)
        if not spec_file.endswith('.yaml'):
            raise ValueError("Module spec file must has extension '.yaml', got '%s'." % spec_file)
        if not force and (Path(working_dir) / spec_file).exists():
            raise FileExistsError("Module spec file '%s' already exists." % spec_file)

    with inject_sys_path(working_dir):
        with ArgumentParserWrapper(entry=entry) as wrapper:
            try:
                if is_file:
                    logger.info("Run py file '%s' to get the args in argparser." % entry)
                    runpy.run_path(Path(working_dir) / entry, run_name='__main__')
                else:
                    logger.info("Run py module '%s' to get the args in argparser." % entry)
                    runpy.run_module(entry, run_name='__main__')
            except ImportError:
                # For ImportError, it could be environment problems, just raise it.
                raise
            except BaseException as e:
                # If the entry is correctly returned, wrapper.succeeded will be True
                # Otherwise it is not run correctly, just raise an exception.
                if not wrapper.succeeded:
                    msg = "Run entry '%s' failed, please make sure it uses argparse correctly." % entry
                    raise RuntimeError(msg) from e
            if not wrapper.succeeded:
                raise ValueError("Entry '%s' doesn't call parser.parse_known_args()." % entry)

            if wrapper.generator.name is None:
                wrapper.generator.set_name(entry)

            wrapper.generator.set_entry(entry)

            if inputs:
                wrapper.generator.update_spec_params(inputs, is_output=False)
            if outputs:
                wrapper.generator.update_spec_params(outputs, is_output=True)

            if module_meta:
                wrapper.generator.update_module_meta(module_meta)

            if target_file:
                wrapper.generator.to_module_entry_file(Path(working_dir) / target_file)
                logger.info("Module entry file '%s' is dumped." % target_file)

            if spec_file:
                wrapper.generator.to_spec_yaml(working_dir, spec_file)
                logger.info("Module spec file '%s' is dumped." % spec_file)


ORIGINAL_ARGUMENT_PARSER = argparse.ArgumentParser
wrapper = None


class _ArgumentParser(argparse.ArgumentParser):
    """This is a class used for generate dsl.module with an existing main code with argparser.

    Usage:
    Replace argparse.ArgumentParser with this _ArgumentParser,
    then when the entry file is called by "python entry.py",
    your code will not be run, a ModuleGenerator will be prepared for generating dsl.module.
    """
    def __init__(self, prog=None, *args, **kwargs):
        """Init the ArgumentParser with spec args."""
        argparse.ArgumentParser = ORIGINAL_ARGUMENT_PARSER
        super().__init__(prog, *args, **kwargs)
        self.injected_generator = ModuleGenerator(description=self.description or self.usage)
        if prog:
            self.injected_generator.set_name(name=prog)
        self.parsed = False

    def add_argument(self, *args, **kwargs):
        """Call add_argument of ArgumentParser and add the argument to spec."""
        # Get the argument.
        result = super().add_argument(*args, **kwargs)

        action = kwargs.get('action')
        # The action help is used for help message, which is useless.
        if action == 'help':
            return result
        # Currently we only support default action 'store', AzureML cannot support others.
        if action and action != 'store':
            logger.warning("Argument action type '%s' of '%s' is not supported now, ignored." % (action, result.dest))
            return result

        # Get meta information of the argument.
        options = result.choices if result.type in {None, str} else []  # Only str type enum is valid.
        param_type = 'Enum' if options else ParamGenerator.mapping.get(result.type, 'String')
        default = result.default if result.default != argparse.SUPPRESS and result.default != [] else None
        optional = not result.required or default is not None

        # Add param to the generator.
        self.injected_generator.add_param(Param(
            name=_to_camel_case(result.dest), type=param_type,
            options=options, description=result.help,
            default=default, optional=optional,
            arg_name=result.dest,
            arg_string=result.option_strings[0],
        ))
        # Return the result to make sure the result is ok.
        return result

    def parse_known_args(self, args=None, namespace=None):
        # Set parsed=True then exit the program.
        # Note that parser.parse_args will also call parse_known_args, so both call are supported.
        self.parsed = True
        sys.exit(0)


class ArgumentParserWrapper:

    def __init__(self, entry):
        self._generator = None
        self._parser = None
        self._entry = entry

    def init_parser_and_inject(self, prog=None, *args, **kwargs):
        """Init an injected _ArgumentParser instance to replace the original argparse.ArgumentParser instance."""
        # Recover argparse.ArgumentParser because _ArgumentParser.__init__ will call argparse.ArgumentParser.__init__,
        # since _ArgumentParser is a subclass of it, if we don't recover, the initialization will fail.

    def __enter__(self):
        # Once using with(), store original ArgumentParser for recovering,
        # then set argparse.ArgumentParser = _Argu
        # Thus when the user code call 'parser = argparse.ArgumentParser()', it actually calls self.__call__()
        global wrapper
        wrapper = self

        class WrapperdArgumentParser(_ArgumentParser):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                wrapper._parser = self
        argparse.ArgumentParser = WrapperdArgumentParser
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        argparse.ArgumentParser = ORIGINAL_ARGUMENT_PARSER

    @property
    def parser(self):
        return self._parser

    @property
    def generator(self):
        return self._parser.injected_generator

    @property
    def succeeded(self):
        return self._parser and self._parser.parsed
