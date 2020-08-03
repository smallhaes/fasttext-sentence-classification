# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
from ._restclients.designer.models import EntityStatus


def _normalize_identifier_name(name):
    import re
    normalized_name = name.lower()
    normalized_name = re.sub(r'[\W_]', ' ', normalized_name)  # No non-word characters
    normalized_name = re.sub(' +', ' ', normalized_name).strip()  # No double spaces, leading or trailing spaces
    if re.match(r'\d', normalized_name):
        normalized_name = 'n' + normalized_name  # No leading digits
    return normalized_name


def _sanitize_python_variable_name(name: str):
    return _normalize_identifier_name(name).replace(' ', '_')


def _get_or_sanitize_python_name(name: str, name_map: dict):
    return name_map[name] if name in name_map.keys() \
        else _sanitize_python_variable_name(name)


def is_float_convertible(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_int_convertible(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def is_bool_string(string):
    if not isinstance(string, str):
        return False
    return string == 'True' or string == 'False'


def int_str_to_pipeline_status(str):
    if str == '0':
        return EntityStatus.active.value
    elif str == '1':
        return EntityStatus.deprecated.value
    elif str == '2':
        return EntityStatus.disabled.value
    else:
        return 'Unknown'


def _unique(elements, key):
    return list({key(element): element for element in elements}.values())


def _is_prod_workspace(workspace):
    return workspace.location != "eastus2euap" and workspace.location != "centraluseuap" and \
        workspace.subscription_id != "4faaaf21-663f-4391-96fd-47197c630979"


def _in_jupyter_nb():
    """Return true if the platform widget is running on is Jupyter notebook, otherwise return false."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ModuleNotFoundError):
        return False  # Probably standard Python interpreter


def _is_json_string_convertible(string):
    try:
        json.loads(string)
    except ValueError:
        return False
    return True
