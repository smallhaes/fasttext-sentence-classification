# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import types
from typing import Sequence, Callable
from inspect import Parameter, Signature


class KwParameter(Parameter):
    """A keyword only parameter with a default value."""
    def __init__(self, name, default, annotation=Parameter.empty, _type='str'):
        super().__init__(name, Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        self._type = _type


def _replace_function_name(func: types.FunctionType, new_name):
    """Return a function with the same body but a new name."""
    try:
        # Use the original code of the function to initialize a new code object for the new function.
        code_template = func.__code__
        # For python>=3.8, it is recommended to use `CodeType.replace`, since the interface is change in py3.8
        # See https://github.com/python/cpython/blob/384621c42f9102e31ba2c47feba144af09c989e5/Objects/codeobject.c#L646
        # The interface has been changed in py3.8, so the CodeType initializing code is invalid.
        # See https://github.com/python/cpython/blob/384621c42f9102e31ba2c47feba144af09c989e5/Objects/codeobject.c#L446
        if hasattr(code_template, 'replace'):
            code = code_template.replace(co_name=new_name)
        else:
            # Before python<3.8, replace is not available, we can only initialize the code as following.
            # https://github.com/python/cpython/blob/v3.7.8/Objects/codeobject.c#L97
            code = types.CodeType(
                code_template.co_argcount,
                code_template.co_kwonlyargcount,
                code_template.co_nlocals,
                code_template.co_stacksize,
                code_template.co_flags,
                code_template.co_code,
                code_template.co_consts,
                code_template.co_names,
                code_template.co_varnames,
                code_template.co_filename,
                new_name,  # Use the new name for the new code object.
                code_template.co_firstlineno,
                code_template.co_lnotab,
                # The following two values are required for closures.
                code_template.co_freevars,
                code_template.co_cellvars,
            )
        # Initialize a new function with the code object and the new name, see the following ref for more details.
        # https://github.com/python/cpython/blob/4901fe274bc82b95dc89bcb3de8802a3dfedab32/Objects/clinic/funcobject.c.h#L30
        return types.FunctionType(
            code,
            globals=func.__globals__,
            name=new_name,
            argdefs=func.__defaults__,
            # Closure must be set to make sure free variables work.
            closure=func.__closure__,
        )
    except BaseException:
        # If the dynamic replacing failed in corner cases, simply set the two fields.
        func.__name__ = func.__qualname__ = new_name
        return func


def _assert_arg_valid(kwargs, keys, func_name):
    """Assert the arg keys are all in keys."""
    for key in kwargs:
        if key not in keys:
            raise TypeError("%s() got an unexpected keyword argument %r, valid keywords: %s." % (
                func_name, key, ', '.join('%r' % key for key in keys)
            ))


def _update_dct_if_not_exist(dst, src):
    """Update the dst dict with the source dict if the key is not in the dst dict."""
    for k, v in src.items():
        if k not in dst:
            dst[k] = v


def create_kw_function_from_parameters(
    func: Callable,
    parameters: Sequence[Parameter],
    func_name: str,
    documentation: str,
    old_to_new_param_name_dict: dict = None
):
    """Create a new keyword only function with provided parameters."""
    if any(p.default == p.empty or p.kind != Parameter.KEYWORD_ONLY for p in parameters):
        raise ValueError("This function only accept keyword only parameters.")
    default_kwargs = {p.name: p.default for p in parameters}

    def f(**kwargs):
        # Update old version kwargs to make function compatibale
        from ._module import _compatible_old_version_params
        kwargs = _compatible_old_version_params(kwargs, old_to_new_param_name_dict)
        # We need to make sure all keys of kwargs are valid.
        _assert_arg_valid(kwargs, default_kwargs, func_name=func_name)
        # We need to put the default args to the kwargs before invoking the original function.
        _update_dct_if_not_exist(kwargs, default_kwargs)
        return func(**kwargs)
    f = _replace_function_name(f, func_name)
    # Set the signature so jupyter notebook could have param hint by calling inspect.signature()
    f.__signature__ = Signature(parameters)
    # Set doc/name/module to make sure help(f) shows following expected result.
    # Expected help(f):
    #
    # Help on function FUNC_NAME:
    # FUNC_NAME(SIGNATURE)
    #     FUNC_DOC
    #
    f.__doc__ = documentation  # Set documentation to update FUNC_DOC in help.
    # Set module = None to avoid showing the sentence `in module 'azureml.pipeline.wrapper._dynamic' in help.`
    # See https://github.com/python/cpython/blob/2145c8c9724287a310bc77a2760d4f1c0ca9eb0c/Lib/pydoc.py#L1757
    f.__module__ = None
    return f


def create_kw_method_from_parameters(
    method: types.MethodType,
    parameters: Sequence[Parameter],
    old_to_new_param_name_dict: dict = None
):
    """Create a new keyword only method with provided parameters."""
    if any(p.default == p.empty or p.kind != Parameter.KEYWORD_ONLY for p in parameters):
        raise ValueError("'%s' only accept keyword only parameters." % create_kw_function_from_parameters)

    default_kwargs = {p.name: p.default for p in parameters}

    # The method_func is the original function defined by user in the class body.
    # Use this function to initialize the new function in the new method.
    method_func = method.__func__

    # As an instance method, the first argument should be positional arg 'self',
    # which is used to put the instance when calling the function using self.new_method(**kwargs).
    # See https://github.com/python/cpython/blob/d9ea5cae1d07e1ee8b03540a9367c26205e0e360/Objects/classobject.c#L465
    def f(self, **kwargs):
        # Update old version kwargs to make function compatibale
        from ._module import _compatible_old_version_params
        kwargs = _compatible_old_version_params(kwargs, old_to_new_param_name_dict)
        # We need to make sure all keys of kwargs are valid.
        _assert_arg_valid(kwargs, default_kwargs, func_name=method_func.__name__)
        # We need to put the default args to the kwargs before invoking the method.
        _update_dct_if_not_exist(kwargs, default_kwargs)
        # method_func(self, **kwargs) equals to self.method(**kwargs).
        return method_func(self, **kwargs)
    f = _replace_function_name(f, method_func.__name__)
    # Keep the attribute of the original method so help() could work well.
    for attr in ['__doc__', '__module__', '__qualname__']:
        setattr(f, attr, getattr(method_func, attr))
    # As a method, the signature of the first argument will be skipped, we need to add one.
    # See https://github.com/python/cpython/blob/384621c42f9102e31ba2c47feba144af09c989e5/Lib/inspect.py#L2245
    f.__signature__ = Signature([Parameter(name='self', kind=Parameter.POSITIONAL_ONLY)] + list(parameters))
    # Initialize a method with the function and the instance.
    # See https://github.com/python/cpython/blob/d9ea5cae1d07e1ee8b03540a9367c26205e0e360/Objects/classobject.c#L210
    new_method = types.MethodType(f, method.__self__)
    return new_method
