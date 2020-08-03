# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml.pipeline.wrapper.dsl._utils import logger, _sanitize_python_class_name
from azureml.pipeline.wrapper.dsl.module import _to_camel_case
from azureml.pipeline.wrapper.dsl._module_spec import BaseModuleSpec, Param, OutputPort, InputPort
from azureml.pipeline.wrapper._module import _sanitize_python_variable_name


ENTRY_TPL = """import sys
import runpy
from enum import Enum
from azureml.pipeline.wrapper import dsl
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor{imports}
{enums}

@dsl.module({dsl_param_dict})
def {func_name}({func_args}
):
    sys.argv = [{sys_argv}
    ]
{append_stmt}
    print(' '.join(sys.argv))
    runpy.run_{module_or_path}('{entry}', run_name='__main__')


if __name__ == '__main__':
    ModuleExecutor({func_name}).execute(sys.argv)
"""


def is_py_file(entry: str):
    """Check whether the file is a python file."""
    return entry.endswith('.py')


class ParamGenerator:
    mapping = {str: 'String', int: 'Int', float: 'Float'}
    reverse_mapping = {v: k.__name__ for k, v in mapping.items()}

    def __init__(self, param: Param):
        self.param = param

    @property
    def type(self):
        return self.param.type

    @property
    def description(self):
        return self.param.description

    @property
    def default(self):
        value = self.param.default if isinstance(self.param, Param) else None
        if value is None:
            if hasattr(self.param, 'optional') and self.param.optional:
                return 'None'
            return None
        if self.type == 'String':
            return "'%s'" % value
        elif self.type == 'Enum':
            if value not in self.param.options:
                value = self.param.options[0]
            return "%s.%s" % (self.enum_class, self.enum_name(value, self.param.options.index(value)))
        return str(value)

    @property
    def var_name(self):
        return _sanitize_python_variable_name(self.param.name) if self.param.arg_name is None else self.param.arg_name

    @property
    def arg_value(self):
        if self.param.type == 'Enum':
            return self.var_name + '.value'
        return 'str(%s)' % self.var_name

    @property
    def arg_type(self):
        if isinstance(self.param, (InputPort, OutputPort)):
            desc_str = 'description="%s"' % self.description if self.description else ''
            key = 'Input' if isinstance(self.param, InputPort) else 'Output'
            return "%sDirectory(%s)" % (key, desc_str)
        if not self.description:
            return self.enum_class if self.type == 'Enum' else self.reverse_mapping[self.type]
        tpl = "EnumParameter(enum={enum_class}, description=\"{description}\")" \
            if self.type == 'Enum' else "{type}Parameter(description=\"{description}\")"
        return tpl.format(type=self.type, enum_class=self.enum_class, description=self.description)

    @property
    def argv(self):
        return ["'%s'" % self.param.arg_string, self.arg_value]

    @property
    def is_optional_argv(self):
        return isinstance(self.param, Param) and self.param.optional is True and self.param.default is None

    @property
    def append_argv_statement(self):
        return """    if %s is not None:\n        sys.argv += [%s]""" % (self.var_name, ', '.join(self.argv))

    @property
    def arg_def(self):
        result = "%s: %s" % (self.var_name, self.arg_type)
        result += ',' if self.default is None else ' = %s,' % self.default
        return result

    @property
    def enum_class(self):
        return 'Enum%s' % _sanitize_python_class_name(self.var_name)

    @staticmethod
    def enum_name(value, idx):
        name = _sanitize_python_variable_name(str(value))
        if name == '':
            name = 'enum%d' % idx
        return name

    @staticmethod
    def enum_value(value):
        return "'%s'" % value

    @property
    def enum_name_def(self):
        return '\n'.join("    %s = %s" % (self.enum_name(option, i), self.enum_value(option))
                         for i, option in enumerate(self.param.options))

    @property
    def enum_def(self):
        return "class {enum_class}(Enum):\n{enum_value_string}\n".format(
            enum_class=self.enum_class, enum_value_string=self.enum_name_def,
        )


class ModuleGenerator:

    def __init__(self, name=None, entry=None, description=None):
        self.params = []
        self.name = None
        if name is not None:
            self.set_name(name)
        self.entry = None
        self.module_or_path = 'path'
        if entry is not None:
            self.set_entry(entry)
        self.description = description
        self._module_meta = {}

    def set_name(self, name):
        if name.endswith('.py'):
            name = name[:-3]
        # Use the last piece as the module name.
        self.name = _to_camel_case(name.split('/')[-1].split('.')[-1])

    def set_entry(self, entry):
        self.entry = entry
        self.module_or_path = 'path' if is_py_file(entry) else 'module'

    def assert_valid(self):
        if self.name is None:
            raise ValueError("The name of a module could not be None.")
        if self.entry is None:
            raise ValueError("The entry of the module '%s' could not be None." % self.name)

    def add_param(self, param: Param):
        self.params.append(ParamGenerator(param))

    def to_module_entry_code(self):
        self.assert_valid()
        keys = [
            'enums', 'imports',
            'entry', 'module_or_path',
            'func_name', 'func_args',
            'sys_argv', 'append_stmt',
            'dsl_param_dict',
        ]
        return ENTRY_TPL.format(**{key: getattr(self, key) for key in keys})

    def to_module_entry_file(self, target='entry.py'):
        with open(target, 'w') as fout:
            fout.write(self.to_module_entry_code())

    @property
    def module_entry_file(self):
        if is_py_file(self.entry):
            return self.entry
        return self.entry.replace('.', '/') + '.py'

    @property
    def spec(self):
        """This spec is directly generated by argument parser arguments,
        it is used to create a module spec without a new entry file.
        """
        params = [param.param for param in self.params if isinstance(param.param, Param)]
        inputs = [param.param for param in self.params if isinstance(param.param, InputPort)]
        outputs = [param.param for param in self.params if isinstance(param.param, OutputPort)]
        args = []
        for param in self.params:
            if not isinstance(param.param, OutputPort) and param.param.optional:
                args.append(param.param.arg_group())
            else:
                args += param.param.arg_group()

        return BaseModuleSpec(
            name=self.name, description=self.description,
            inputs=inputs, outputs=outputs, params=params,
            args=args,
            command=['python', self.module_entry_file],
        )

    @property
    def spec_dict(self):
        return self.spec.spec_dict

    def to_spec_yaml(self, folder, spec_file='spec.yaml'):
        self.assert_valid()
        self.spec.dump_module_spec_to_folder(folder, spec_file=spec_file)

    def has_type(self, type):
        return any(param.type == type for param in self.params)

    def has_import_type(self, type):
        return any(param.type == type and param.description is not None for param in self.params)

    def has_input(self):
        return any(isinstance(param.param, InputPort) for param in self.params)

    def has_output(self):
        return any(isinstance(param.param, OutputPort) for param in self.params)

    @property
    def enums(self):
        return '\n\n' + '\n\n'.join(param.enum_def for param in self.params if param.type == 'Enum') \
            if self.has_type('Enum') else ''

    @property
    def func_name(self):
        return _sanitize_python_variable_name(self.name)

    @property
    def imports(self):
        keys = ['Enum'] + list(ParamGenerator.reverse_mapping)
        param_imports = [''] + ['%sParameter' % key for key in keys if self.has_import_type(key)]
        if self.has_input():
            param_imports.append('InputDirectory')
        if self.has_output():
            param_imports.append('OutputDirectory')
        return ', '.join(param_imports)

    @property
    def func_args(self):
        items = [''] + [param.arg_def for param in self.params if param.default is None] + \
                [param.arg_def for param in self.params if param.default is not None]
        return '\n    '.join(items)

    @property
    def sys_argv(self):
        items = ['', "'%s'," % self.entry] + [
            ', '.join(param.argv) + ',' for param in self.params if not param.is_optional_argv
        ]
        return '\n        '.join(items)

    @property
    def append_stmt(self):
        return '\n'.join(param.append_argv_statement for param in self.params if param.is_optional_argv)

    @property
    def dsl_param_dict(self):
        meta = self.module_meta
        if not meta:
            return ''
        items = [''] + ['%s=%r,' % (k, v) for k, v in meta.items()]
        return '\n    '.join(items) + '\n'

    def update_spec_param(self, key, is_output=False):
        target = None
        key = key
        for param in self.params:
            if param.var_name == key:
                target = param
                break
        if not target:
            logger.warning(f"{key} not found in params.")
            return
        param = target.param
        if is_output:
            target.param = OutputPort(
                name=param.name, type="AnyDirectory", description=param.description,
                arg_string=param.arg_string,
            )
        else:
            target.param = InputPort(
                name=param.name, type="AnyDirectory",
                description=target.description, optional=param.optional,
                arg_string=param.arg_string,
            )

    def update_spec_params(self, keys, is_output=False):
        for key in keys:
            self.update_spec_param(key, is_output)

    @property
    def module_meta(self):
        meta = {**self._module_meta}
        if self.description and 'description' not in meta:
            meta['description'] = self.description
        if self.name and 'name' not in meta:
            meta['name'] = self.name
        return meta

    def update_module_meta(self, module_meta):
        for k, v in module_meta.items():
            if v is not None:
                self._module_meta[k] = v
