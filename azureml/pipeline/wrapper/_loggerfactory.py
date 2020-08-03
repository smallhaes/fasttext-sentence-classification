# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import logging.handlers
import uuid
import json
import os
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Callable, Any, TypeVar

_PUBLIC_API = 'PublicApi'

COMPONENT_NAME = 'azureml.pipeline.wrapper'
session_id = 'l_' + str(uuid.uuid4())
default_custom_dimensions = {}
ActivityLoggerAdapter = None

try:
    from azureml.telemetry import get_telemetry_log_handler
    from azureml.telemetry.activity import ActivityType, ActivityLoggerAdapter, ActivityCompletionStatus
    from azureml.telemetry.logging_handler import AppInsightsLoggingHandler
    from azureml._base_sdk_common import _ClientSessionId
    session_id = _ClientSessionId

    telemetry_enabled = True
    DEFAULT_ACTIVITY_TYPE = ActivityType.INTERNALCALL
except Exception:
    telemetry_enabled = False
    DEFAULT_ACTIVITY_TYPE = 'InternalCall'


class _LoggerFactory:
    _core_version = None
    _dataprep_version = None

    @staticmethod
    def get_logger(name, verbosity=logging.DEBUG):
        logger = logging.getLogger(__name__).getChild(name)
        logger.propagate = False
        logger.setLevel(verbosity)
        if telemetry_enabled:
            if not _LoggerFactory._found_handler(logger, AppInsightsLoggingHandler):
                logger.addHandler(get_telemetry_log_handler(component_name=COMPONENT_NAME))

        return logger

    @staticmethod
    def track_activity(logger, activity_name, activity_full_name=None,
                       activity_type=DEFAULT_ACTIVITY_TYPE, input_custom_dimensions=None):
        _LoggerFactory._get_version_info()

        if input_custom_dimensions is not None:
            custom_dimensions = default_custom_dimensions.copy()
            custom_dimensions.update(input_custom_dimensions)
        else:
            custom_dimensions = default_custom_dimensions
        custom_dimensions.update({
            'source': COMPONENT_NAME,
            'version': _LoggerFactory._core_version,
            'dataprepVersion': _LoggerFactory._dataprep_version
        })
        if telemetry_enabled:
            return _log_activity(logger, activity_name, activity_full_name, activity_type, custom_dimensions)
        else:
            return _log_local_only(logger, activity_name, activity_full_name, activity_type, custom_dimensions)

    @staticmethod
    def trace(logger, message, custom_dimensions=None, adhere_custom_dimensions=True):
        # Put custom_dimensions inside logger for future use
        if adhere_custom_dimensions:
            logger.custom_dimensions = custom_dimensions
        payload = dict(pid=os.getpid())
        payload.update(custom_dimensions or {})
        payload['version'] = _LoggerFactory._core_version
        payload['source'] = COMPONENT_NAME

        if ActivityLoggerAdapter:
            activity_logger = ActivityLoggerAdapter(logger, payload)
            activity_logger.info(message)
        else:
            logger.info('Message: {}\nPayload: {}'.format(message, json.dumps(payload)))

    @staticmethod
    def _found_handler(logger, handler_type):
        for log_handler in logger.handlers:
            if isinstance(log_handler, handler_type):
                return True
        return False

    @staticmethod
    def _get_version_info():
        if _LoggerFactory._core_version is not None and _LoggerFactory._dataprep_version is not None:
            return

        core_ver = _get_package_version('azureml-core')
        if core_ver is None:
            # only fallback when the approach above fails, as azureml.core.VERSION has no patch version segment
            try:
                from azureml.core import VERSION as core_ver
            except Exception:
                core_ver = ''
        _LoggerFactory._core_version = core_ver

        dprep_ver = _get_package_version('azureml-dataprep')
        if dprep_ver is None:
            try:
                from azureml.dataprep import __version__ as dprep_ver
            except Exception:
                # data-prep may not be installed
                dprep_ver = ''
        _LoggerFactory._dataprep_version = dprep_ver


# hint vscode intellisense
_TFunc = TypeVar("_TFunc", bound=Callable[..., Any])


def track(get_logger, custom_dimensions=None, activity_type=DEFAULT_ACTIVITY_TYPE, activity_name=None):
    def monitor(func: _TFunc) -> _TFunc:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            _activity_name = activity_name if activity_name is not None else func.__name__
            _activity_full_name = f'{func.__module__}.{func.__qualname__}'
            with _LoggerFactory.track_activity(logger, _activity_name, _activity_full_name,
                                               activity_type, custom_dimensions) as al:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    al.activity_info['exception_type'] = type(e).__name__
                    al.activity_info['exception_detail'] = json.dumps(_get_exception_detail(e))
                    raise

        return wrapper

    return monitor


@contextmanager
def _log_activity(logger, activity_name, activity_full_name=None,
                  activity_type=ActivityType.INTERNALCALL, custom_dimensions=None):
    activity_info = dict(activity_id=str(uuid.uuid4()), activity_name=activity_name, activity_type=activity_type)
    if activity_full_name is not None:
        activity_info.update({'activity_full_name': activity_full_name})

    custom_dimensions = custom_dimensions or {}
    activity_info.update(custom_dimensions)

    start_time = datetime.utcnow()
    completion_status = ActivityCompletionStatus.SUCCESS

    message = "ActivityStarted, {}".format(activity_name)
    activityLogger = ActivityLoggerAdapter(logger, activity_info)
    activityLogger.info(message)
    exception = None

    try:
        yield activityLogger
    except Exception as e:
        exception = e
        completion_status = ActivityCompletionStatus.FAILURE
        raise
    finally:
        end_time = datetime.utcnow()
        duration_ms = round((end_time - start_time).total_seconds() * 1000, 2)

        # Add additional dimensions from logger and clear it after use.
        if hasattr(logger, 'custom_dimensions'):
            activityLogger.activity_info.update(logger.custom_dimensions or {})
            delattr(logger, 'custom_dimensions')

        activityLogger.activity_info["completionStatus"] = completion_status
        activityLogger.activity_info["durationMs"] = duration_ms
        message = "ActivityCompleted: Activity={}, HowEnded={}, Duration={} [ms]".format(
            activity_name, completion_status, duration_ms)
        if exception:
            message += ", Exception={}".format(type(exception).__name__)
            activityLogger.error(message)
        else:
            activityLogger.info(message)


@contextmanager
def _log_local_only(logger, activity_name, activity_full_name,
                    activity_type, custom_dimensions):
    activity_info = dict(activity_id=str(uuid.uuid4()), activity_name=activity_name, activity_type=activity_type)
    if activity_full_name is not None:
        activity_info.update({'activity_full_name': activity_full_name})

    custom_dimensions = custom_dimensions or {}
    activity_info.update(custom_dimensions)

    start_time = datetime.utcnow()
    completion_status = 'Success'

    message = 'ActivityStarted, {}'.format(activity_name)
    logger.info(message)
    exception = None

    try:
        yield logger
    except Exception as e:
        exception = e
        completion_status = 'Failure'
        raise
    finally:
        end_time = datetime.utcnow()
        duration_ms = round((end_time - start_time).total_seconds() * 1000, 2)

        if hasattr(logger, 'custom_dimensions'):
            logger.activity_info.update(logger.custom_dimensions)
            delattr(logger, 'custom_dimensions')

        custom_dimensions['completionStatus'] = completion_status
        custom_dimensions['durationMs'] = duration_ms
        message = '{} | ActivityCompleted: Activity={}, HowEnded={}, Duration={} [ms], Info = {}'.format(
            start_time, activity_name, completion_status, duration_ms, repr(activity_info))
        if exception:
            message += ', Exception={}; {}'.format(type(exception).__name__, str(exception))
            logger.error(message)
        else:
            logger.info(message)


def _get_package_version(package_name):
    import pkg_resources
    try:
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        # Azure CLI exception loads azureml-* package in a special way which makes get_distribution not working
        try:
            all_packages = pkg_resources.AvailableDistributions()  # scan sys.path
            for name in all_packages:
                if name == package_name:
                    return all_packages[name][0].version
        except Exception:
            # In case this approach is not working neither
            return None


def _get_exception_detail(e: Exception):
    exception_detail = {}
    # azureml._restclient.modules.error_response.ErrorResponseException
    if hasattr(e, 'response'):
        exception_detail['http_status_code'] = e.response.status_code
        exception_detail['http_error_message'] = e.message
    # msrest.exceptions.ClientRequestError & azureml._common.exception.AzureMLException
    if hasattr(e, 'inner_exception') and e.inner_exception is not None:
        exception_detail['inner_exception_type'] = type(e.inner_exception).__name__
        if hasattr(e.inner_exception, 'message'):
            exception_detail['inner_exception_error_message'] = e.inner_exception.message
    # azureml._common.exception.AzureMLException
    if hasattr(e, '_error_code'):
        exception_detail['error_code'] = e._error_code
    if hasattr(e, 'message'):
        exception_detail['error_message'] = e.message
    # Other exceptions
    else:
        exception_detail['error_message'] = str(e)
    return exception_detail
