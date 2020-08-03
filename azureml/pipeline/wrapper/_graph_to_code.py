# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os
import re
from enum import Enum
from azureml.core import Workspace
from msrest.exceptions import HttpOperationError
from azureml.pipeline.wrapper._restclients.service_caller import DesignerServiceCaller
from ._loggerfactory import _LoggerFactory, track

_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


class WrappedHttpOperationException(Exception):
    """
    This class wrapped http operation exception error message.

    :param http_operation_exception: The original exception
    :type http_operation_exception: HttpOperationError
    """

    def __init__(self, http_operation_excpetion: HttpOperationError, **kwargs):
        error_code = http_operation_excpetion.response.status_code
        error_detail = http_operation_excpetion.response.text
        if error_code < 500:
            exception_message = "User Error: {}. Details: {}".format(error_code, error_detail)
        else:
            exception_message = "System Error: {}. Details: {}".format(error_code, error_detail)
        Exception.__init__(self, exception_message, **kwargs)


class _ExportedFormat(Enum):
    Python = 0
    JupyterNotebook = 1


def wrapDesignerException(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HttpOperationError as e:
            raise WrappedHttpOperationException(e)
    return wrapper


def _find_content_disposition_filename(headers):
    if 'Content-Disposition' in headers.keys():
        fname = re.findall("filename=(.+);", headers['Content-Disposition'])
        if len(fname) > 0:
            return fname[0]
    return None


def _get_base_name_without_ext(filename):
    base_name = os.path.basename(filename)
    return os.path.splitext(base_name)[0]


def _get_default_output_file_name(export_format: _ExportedFormat, draft_id=None, run_id=None):
    prefix = "run_{}".format(run_id) if draft_id is None else "draft_{}".format(draft_id)
    postfix = ""
    if export_format == "Python":
        ext = ".py"
    elif export_format == "JupyterNotebook":
        ext = ".ipynb"
    exist_count = 0
    default_name = prefix + postfix + ext
    while os.path.isfile(default_name):
        exist_count += 1
        postfix = "({})".format(exist_count)
        default_name = prefix + postfix + ext
    return default_name


def _save_graph_sdk_code(result, path, default_output_file_name):
    """
    save graph sdk code result according to response and user specified path and file name
    return the code saved full path
    """
    saved_code_path = None
    response_filename = _find_content_disposition_filename(result.headers)

    if response_filename is None or not response_filename.endswith(".zip"):
        saved_code_path = os.path.join(path, default_output_file_name)
        with open(saved_code_path, 'wb') as f:
            f.write(result.content)
    else:
        # first download the zip file
        zipFileFullName = os.path.join(path, response_filename)
        with open(zipFileFullName, 'wb') as f:
            f.write(result.content)
        internal_folder = _get_base_name_without_ext(response_filename)
        internal_folder = os.path.join(path, internal_folder)
        saved_code_path = internal_folder
        exist_count = 0
        while os.path.isdir(saved_code_path):
            exist_count += 1
            postfix = "({})".format(exist_count)
            saved_code_path = internal_folder + postfix
        from zipfile import ZipFile
        with ZipFile(zipFileFullName, 'r') as zip:
            zip.extractall(saved_code_path)
        os.remove(zipFileFullName)

    return saved_code_path


def _parse_designer_url(url):
    """
    assume there are only four kinds of valid url
    draft: /Normal/{draft_id}?wsid=/subscriptions/{sub_id}/resourcegroups/{rg}/workspaces/{ws}
    run: /pipelineruns/id/{exp_id}/{run_id}?wsid=/subscriptions/{sub_id}/resourcegroups/{rg}/workspaces/{ws}
    run: /pipelineruns/{exp_name}/{run_id}?wsid=/subscriptions/{sub_id}/resourcegroups/{rg}/workspaces/{ws}
    run: /runs/{run_id}?wsid=/subscriptions/{run_id}/resourcegroups/{rg}/workspaces/{ws}
    """
    entries = re.split(r'[/&?]', url)
    subscription_id = None
    resource_group = None
    workspace_name = None
    draft_id = None
    run_id = None

    def _get_entry_index_value(entries, index):
        if len(entries) >= index + 1:
            return entries[index]
        else:
            return None

    for i, entry in enumerate(entries):
        if entry == "runs":
            run_id = _get_entry_index_value(entries, i + 1)
        elif entry == "pipelineruns":
            if _get_entry_index_value(entries, i + 1) == "id":
                run_id = _get_entry_index_value(entries, i + 3)
            else:
                run_id = _get_entry_index_value(entries, i + 2)
        elif entry == "Normal":
            draft_id = _get_entry_index_value(entries, i + 1)
        elif entry == "subscriptions":
            subscription_id = _get_entry_index_value(entries, i + 1)
        elif entry == "resourcegroups":
            resource_group = _get_entry_index_value(entries, i + 1)
        elif entry == "workspaces":
            workspace_name = _get_entry_index_value(entries, i + 1)

    if draft_id is None and run_id is None:
        raise ValueError("Invalid url. No draft_id or run_id found.")

    if subscription_id is None or resource_group is None or workspace_name is None:
        raise ValueError("Invalid url. No subscription_id, resource_group or workspace_name found")

    return subscription_id, resource_group, workspace_name, draft_id, run_id


@track(_get_logger, activity_name="_export_pipeline_draft_to_code")
@wrapDesignerException
def _export_pipeline_draft_to_code(workspace: Workspace, draft_id: str, path: str,
                                   export_format: _ExportedFormat):
    service_caller = DesignerServiceCaller(workspace)
    result = service_caller.get_pipeline_draft_sdk_code(draft_id=draft_id, target_code=export_format)

    default_out_file_name = _get_default_output_file_name(export_format=export_format, draft_id=draft_id)
    return _save_graph_sdk_code(result, path, default_out_file_name)


@track(_get_logger, activity_name="_export_pipeline_run_to_code")
@wrapDesignerException
def _export_pipeline_run_to_code(workspace: Workspace, pipeline_run_id: str, path: str,
                                 export_format: _ExportedFormat, experiment_name: str, experiment_id: str):
    service_caller = DesignerServiceCaller(workspace)
    result = service_caller.get_pipeline_run_sdk_code(pipeline_run_id=pipeline_run_id, target_code=export_format,
                                                      experiment_id=experiment_id, experiment_name=experiment_name)

    default_out_file_name = _get_default_output_file_name(export_format=export_format, run_id=pipeline_run_id)
    return _save_graph_sdk_code(result, path, default_out_file_name)
