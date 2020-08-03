# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains classes for creating and managing resusable computational units of an Azure Machine Learning pipeline.

Modules allow you to create computational units in a :class:`azureml.pipeline.wrapper.Pipeline`, which can have
inputs, outputs, and rely on parameters and an environment configuration to operate.

Modules are designed to be reused in several pipelines and can evolve to adapt a specific computation logic
to different use cases. A step in a pipeline can be used in fast iterations to improve an algorithm,
and once the goal is achieved, the algorithm is usually published as a module to enable reuse.
"""
import os
import re
import types
import importlib
import tempfile
from uuid import UUID
from typing import Any, Mapping, List, Callable
from pathlib import Path
import json
import time
import uuid
import inspect
import warnings

from azureml.core import Workspace, Datastore, ScriptRunConfig
from azureml.core.experiment import Experiment
from azureml.core.run import Run
from azureml.core.runconfig import RunConfiguration
from azureml.data.abstract_dataset import AbstractDataset
from azureml.data.abstract_datastore import AbstractDatastore
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data.data_reference import DataReference
from azureml.exceptions._azureml_exception import UserErrorException

from ._module_validator import ModuleValidator, ValidationError
from ._module_dto import ModuleDto, INPUTS, OUTPUTS, PARAMETERS
from ._dynamic import KwParameter, create_kw_method_from_parameters
from ._module_registration import _load_anonymous_module, _register_module_from_yaml
from ._module_run_helper import _module_run
from ._module_run_helper import _get_module_snapshot
from ._loggerfactory import _LoggerFactory, _PUBLIC_API, track
from ._pipeline_parameters import PipelineParameter
from ._pipeline_data import PipelineData
from ._restclients.service_caller import DesignerServiceCaller
from ._utils import _sanitize_python_variable_name, _get_or_sanitize_python_name
from ._dataset import _GlobalDataset
from ._telemetry import _get_telemetry_value_from_workspace
from ._run_settings import _RunSettings, _K8sRunSettings, set_compute_field
from .debug._constants import DATA_REF_PREFIX

USE_STRUCTURED_ARGUMENTS = 'USE_STRUCTURED_ARGUMENTS'

_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


def _full_name(namespace: str, name: str):
    return namespace + "://" + name


def _get_module_yaml(workspace: Workspace, module_version_id: str):
    """
    Display yaml of module

    :return: yaml of module
    :rtype: str
    """
    try:
        service_caller = DesignerServiceCaller(workspace)
        result = service_caller.get_module_yaml_by_id(module_version_id)
        return str(result)
    except Exception as e:
        warnings.warn("get module yaml meet exception : %s" % str(e))
        return None


def _is_uuid(str):
    try:
        UUID(hex=str)
    except ValueError:
        return False
    return True


def _refine_batch_load_input(ids, identifiers, workspace_name):
    """
    refine batch load input:
        1.replace None value with empty list
        2.standardized tuple length to 3

    :param ids: module_version_ids
    :type List[str]
    :param identifiers: (name,namespace,version) list
    :type List[tuple]
    :param workspace_name: default namespace to fill
    :type str
    :return: input after refined
    :rtype: List[str], List[tuple]
    """
    _ids = [] if ids is None else ids
    _identifiers = []

    badly_formed_id = [_id for _id in _ids if not _is_uuid(_id)]
    if len(badly_formed_id) > 0:
        raise UserErrorException('Badly formed module_version_id found, '
                                 'expected hexadecimal guid, error list {0}'.format(badly_formed_id))

    if identifiers is not None:
        for item in identifiers:
            if isinstance(item, tuple):
                if len(item) > 3:
                    raise UserErrorException('Ambiguous identifier tuple found, '
                                             f'excepted tuple length <= 3, actually {item}')
                while len(item) < 3:
                    item += (None,)
                _identifiers.append(item)
            else:
                _identifiers.append((item, workspace_name, None))
    return _ids, _identifiers


def _refine_batch_load_output(module_dtos, ids, identifiers, workspace_name):
    """
    copy result for duplicate module_version_id
    refine result order

    :param module_dtos: origin result list
    :type List[azureml.pipeline.wrapper._module_dto.ModuleDto]
    :param ids: module_version_ids
    :type List[str]
    :param identifiers: (name,namespace,version) list
    :type List[tuple]
    :return: refined output and filed module version ids and identifiers
    :rtype: List[azureml.pipeline.wrapper._module_dto.ModuleDto], List[str], List[tuple]
    """
    id_set = set(ids)
    id_dto_dict = {module_dto.module_version_id: module_dto
                   for module_dto in module_dtos
                   if module_dto.module_version_id in id_set}
    idf_dto_dict = {_idf: _dto for _dto in module_dtos
                    for _idf in _dto.get_refined_module_dto_identifiers(workspace_name)}

    failed_ids = []
    failed_identifiers = []
    refined_output = []
    for _id in ids:
        if _id in id_dto_dict.keys():
            refined_output.append(id_dto_dict[_id])
        else:
            failed_ids.append(_id)

    for _idf in identifiers:
        if _idf in idf_dto_dict.keys():
            refined_output.append(idf_dto_dict[_idf])
        else:
            failed_identifiers.append(_idf)
    return refined_output, failed_ids, failed_identifiers


class _InputBuilder(object):
    AVAILABLE_MODE = ['mount', 'download']

    def __init__(self, dset, name: str, mode='mount', owner=None):
        self._dset = dset
        self._name = name
        self._mode = mode
        self._owner: Module = owner

    @track(_get_logger, activity_type=_PUBLIC_API)
    def configure(self, mode='mount'):
        """
        Use this method to configure the input.

        :param mode: The mode that will be used for this input. Available options are
            'mount' and 'download'.
        :type mode: str
        """
        if mode not in self.AVAILABLE_MODE:
            raise UserErrorException(f'Invalid mode: {mode}')

        if self._owner is not None:
            self._owner._specify_input_mode = True
        self._mode = mode

    @property
    def name(self):
        """
        Name of the input.

        :return: Name.
        :rtype: str
        """
        return self._name

    @property
    def port_name(self):
        """
        The output display name
        """
        return re.sub(pattern='_', repl=' ', string=self._name).capitalize()

    @property
    def mode(self):
        """
        Mode of the input.

        :return: Mode.
        :rtype: str
        """
        return self._mode

    @property
    def dset(self):
        return self._dset

    def _is_dset_data_source(self):
        """
        indicates whether the internal dset is a real data source
        """
        if isinstance(self.dset, _InputBuilder):
            return self.dset._is_dset_data_source()
        else:
            return self.dset is not None and not isinstance(self.dset, _OutputBuilder) \
                and not isinstance(self.dset, PipelineData)

    def _get_internal_data_source(self):
        """
        get the dset iterativly until the dset is not an _InputBuilder
        """
        if isinstance(self.dset, _InputBuilder):
            return self.dset._get_internal_data_source()
        else:
            return self.dset

    def build(self):
        from ._pipeline import Pipeline
        if isinstance(self._dset, PipelineParameter):
            return self._dset
        if isinstance(self._dset, AbstractDataset):
            if self._mode == 'mount':
                return self._dset.as_named_input(self._name).as_mount()
            elif self._mode == 'download':
                return self._dset.as_named_input(self._name).as_download()
        elif isinstance(self._dset, _OutputBuilder):
            return self._dset.last_build
        elif isinstance(self._dset, _InputBuilder):
            # usually, the _dset should always comes from a source or a output
            # the _dest may be _InputBuilder only to describe that
            # the destination comes from a specific subgraph dummy input port
            return self._dset.build()
        elif isinstance(self._dset, Module) or isinstance(self._dset, Pipeline):
            output_len = len(self._dset.outputs.values()) \
                if self._dset.outputs is not None else 0
            if output_len != 1:
                raise UserErrorException('{0} output(s) found of specified module/pipeline "{1}",'
                                         ' exactly 1 output required.'.format(output_len, self._dset.name))
            self._dset = list(self._dset.outputs.values())[0]
            return self._dset.last_build
        elif isinstance(self._dset, _AttrDict):
            output_len = len(self._dset.values())
            if output_len != 1:
                raise UserErrorException('{0} output(s) found of specified outputs,'
                                         ' exactly 1 output required.'.format(output_len))
            self._dset = list(self._dset.values())[0]
            return self._dset.last_build
        else:
            return self._dset


class _OutputBuilder(object):
    AVAILABLE_MODE = ['mount', 'upload']

    def __init__(self, name: str, datastore=None, output_mode='mount', port_name=None,
                 owner=None):
        self._datastore = datastore
        self._name = name
        self._output_mode = output_mode
        self._last_build = None
        self._port_name = port_name
        self._owner: Module = owner

    @track(_get_logger, activity_type=_PUBLIC_API)
    def configure(self, datastore=None, output_mode='mount'):
        """
        Use this method to configure the output.

        :param datastore: The datastore that will be used to construct PipelineData.
        :type datastore: azureml.core.datastore.Datastore
        :param output_mode: Specifies whether the producing step will use "upload" or "mount"
            method to access the data.
        :type output_mode: str
        """
        if datastore is not None:
            if not isinstance(datastore, AbstractDatastore):
                raise UserErrorException(
                    'Invalid datastore type. Use azureml.core.Datastore for datastore construction.')
            self._datastore = datastore
            if self._owner is not None:
                self._owner._specify_output_datastore = True

        if output_mode != 'mount':
            if output_mode not in self.AVAILABLE_MODE:
                raise UserErrorException(f'Invalid mode: {output_mode}')

            self._output_mode = output_mode
            if self._owner is not None:
                self._owner._specify_output_mode = True

    @property
    def datastore(self):
        return self._datastore

    @property
    def output_mode(self):
        """
        Output mode that will be used to construct PipelineData.

        :return: Output mode.
        :rtype: str
        """
        return self._output_mode

    @property
    def port_name(self):
        """
        The output display name
        """
        return self._port_name

    @property
    def module_instance_id(self):
        """
        Specifies which module instance build this Output
        """
        return self._owner._instance_id

    @property
    def last_build(self):
        return self._last_build

    def build(self, producer=None, default_datastore=None):
        if self._datastore is None:
            self._datastore = default_datastore
        if self._datastore is None:
            raise ValueError("datastore is required")

        self._last_build = PipelineData(self._port_name, datastore=self._datastore, output_mode=self._output_mode)
        self._last_build._set_producer(producer)
        self._last_build._set_port_name(self._port_name)
        return self._last_build


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            ue = UserErrorException(f'Key {name} not found')
            raise ue

    def __setattr__(self, name, value):
        self[name] = value

    # For Jupyter Notebook auto-completion
    def __dir__(self):
        return super().__dir__() + [str(k) for k in self.keys()]

    def __deepcopy__(self, memodict=None):
        attr_dict = _AttrDict()
        for key in self.keys():
            attr_dict.__setattr__(key, self.__getattr__(key))
        return attr_dict


def _compatible_old_version_params(init_params, old_to_new_param_name_dict):
    """
    Make init_params compatible since the rules
        that generate the default argument name have changed

    :param init_params: the init params of function given by user
    :type init_params: dict[str, str]
    :param old_to_new_param_name_dict: a dict to get parameter's
        new `argument_name` by the old one, since the rule of
         generate argument_name has changed. The dict like:
        {'inputs':{'input_path':'input_name_defined_by_myself'}, 'outputs':{}, parameters:{}}
    :type old_to_new_param_name_dict: dict[str, dict]
    :return: compatible_params by replace the old param keys in `init_param` to new keys
    :rtype: dict[str, str]
    """
    if old_to_new_param_name_dict is None or len(old_to_new_param_name_dict) is 0:
        return init_params
    all_param_name_dict = dict(old_to_new_param_name_dict[INPUTS].items() |
                               old_to_new_param_name_dict[PARAMETERS].items() |
                               old_to_new_param_name_dict[OUTPUTS].items())
    compatible_params = {}
    for k, v in init_params.items():
        if k in all_param_name_dict.keys():
            new_key = all_param_name_dict[k]
            if k != new_key:
                warning_str = "The old style parameter name '{0}' will be deprecated " \
                    "in the future. Please use the new parameter name '{1}'.".format(k, new_key)
                warnings.warn(warning_str)
                k = new_key
        compatible_params[k] = v
    return compatible_params


class _ModuleLoadSource(object):

    UNKNOWN = 'unknown'
    REGISTERED = 'registered'
    FROM_YAML = 'from_yaml'
    FROM_FUNC = 'from_func'


class Module(object):
    """
    Define an operational unit that can be used to produce a functional pipeline, which consists of a series of
    `azureml.pipeline.wrapper.Module` nodes.

    Note that you should not use the constructor yourself. Use :meth:`azureml.pipeline.wrapper.Module.load`
    and related methods to acquire the needed `azureml.pipeline.wrapper.Module`.

    .. remarks::

        This main functionality of Module class resides at where we call "module function". A "module function" is
        essentially a function that you can call in Python code, which has parameters and return value that mimics the
        module definition in Azure Machine Learning.

        The following example shows how to create a pipeline using publish methods of the
        :class:`azureml.pipeline.wrapper.Module` class:

        .. code-block:: python

            # Suppose we have a workspace as 'ws'
            input1 = Dataset.get_by_name(ws, name='dset1')
            input2 = Dataset.get_by_name(ws, name='dset2')

            join_data_module_func = Module.load(ws, namespace='azureml', name='Join Data')

            # join_data_module_func is a dynamic-generated function, which has the signature of
            # the actual inputs & parameters of the "Join Data" module.

            join_data = join_data_module_func(
                dataset1=input1,
                dataset2=input2,
                comma_separated_case_sensitive_names_of_join_key_columns_for_l=
                    "{\"KeepInputDataOrder\":true,\"ColumnNames\":[\"MovieId\"]}",
                comma_separated_case_sensitive_names_of_join_key_columns_for_r=
                    "{\"KeepInputDataOrder\":true,\"ColumnNames\":[\"Movie ID\"]}",
                match_case="True",
                join_type="Inner Join",
                keep_right_key_columns_in_joined_table="True"
            )
            # join_data is now a `azureml.pipeline.wrapper.Module` instace.
            # Note that function parameters are optional, you can just use ejoin_module_func()
            and set the parameters afterwards.

            remove_duplicate_rows_module_func = Module.load(ws, namespace='azureml', name='Remove Duplicate Rows')

            remove_duplicate_rows = remove_duplicate_rows_module_func(
                dataset=join_data.outputs.result_dataset, # Note that we can directly use outputs of previous modules.
                key_column_selection_filter_expression=
                    "{\"KeepInputDataOrder\":true,\"ColumnNames\":[\"Movie Name\", \"UserId\"]}",
                retain_first_duplicate_row = "True"
            )

            # Use `azureml.pipeline.wrapper.Pipeline`
            pipeline = Pipeline(
                nodes=[join_data, remove_duplicate_rows],
                outputs={**remove_duplicate_rows.outputs},
            )

            # Submit the run
            pipeline.submit(ws, experiment_name='some_experiment', default_compute_target='some_compute_target')

        Note that we don't support DataReference as input yet. To use a DataReference, you need to
        first register it as a dataset. The following example demonstrates this work-around:

        .. code-block:: python

            try:
                dset = Dataset.get_by_name(ws, 'Automobile_price_data_(Raw)')
            except Exception:
                global_datastore = Datastore(ws, name="azureml_globaldatasets")
                dset = Dataset.File.from_files(global_datastore.path('GenericCSV/Automobile_price_data_(Raw)'))
                dset.register(workspace=ws,
                            name='Automobile_price_data_(Raw)',
                            create_new_version=True)
                dset = Dataset.get_by_name(ws, 'Automobile_price_data_(Raw)')
            blob_input_data = dset
            module = some_module_func(input=blob_input_data)

    For more information about modules, see:

    * `What's an azure ml module <https://github.com/Azure/DesignerPrivatePreviewFeatures>`_

    * `Define a module using module specs <https://aka.ms/azureml-module-specs>`_
    """
    def __init__(self, _workspace: Workspace, _module_dto: ModuleDto, _init_params: Mapping[str, str],
                 _module_name: str, _load_source: str):
        """
        :param _workspace: (Internal use only.) The workspace object this module will belong to.
        :type _workspace: azureml.core.Workspace
        :param _module_dto: (Internal use only.) The ModuleDto object.
        :type _module_dto: azureml.pipeline.wrapper._module_dto
        :param _init_params: (Internal use only.) The init params will be used to initialize inputs and parameters.
        :type _init_params: dict
        """
        self._module_name = _module_name
        self._workspace = _workspace
        self._module_dto = _module_dto
        self._load_source = _load_source
        if "://" in self._module_name:
            self._namespace, self._short_name = self._module_name.split("://")
        else:
            self._namespace = self._module_dto.namespace
            self._short_name = self._module_name
        self.__doc__ = self._module_dto.description if self._module_dto.description else ""
        self._param_python_name_dict = _module_dto.get_module_param_python_name_dict()
        self._old_to_new_param_name_dict = _module_dto.get_old_to_new_param_name_dict()

        init_params = _compatible_old_version_params(dict(_init_params),
                                                     self._old_to_new_param_name_dict)

        # generate a id for current module instance
        self._instance_id = str(uuid.uuid4())

        # Inputs
        interface: StructuredInterface = self._module_dto.module_entity.structured_interface
        self._interface_inputs: List[StructuredInterfaceInput] = interface.inputs
        self._pythonic_name_to_input_map = {
            _get_or_sanitize_python_name(i.name, self._param_python_name_dict[INPUTS]):
                i.name for i in self._interface_inputs
        }

        input_builder_map = {k: _InputBuilder(v, k, owner=self) for k, v in init_params.items()
                             if k in self._pythonic_name_to_input_map.keys()}
        self._inputs: _AttrDict = _AttrDict(input_builder_map)

        # Parameters
        self._interface_parameters: List[StructuredInterfaceParameter] = interface.parameters
        self._pythonic_name_to_parameter_map = {
            _get_or_sanitize_python_name(parameter.name, self._param_python_name_dict[PARAMETERS]):
                parameter.name for parameter in self._interface_parameters
        }

        self._parameter_params = {k: v for k, v in init_params.items()
                                  if k in self._pythonic_name_to_parameter_map.keys()}

        # Outputs
        self._interface_outputs = interface.outputs
        self._pythonic_name_to_output_map = {
            _get_or_sanitize_python_name(i.name, self._param_python_name_dict[OUTPUTS]):
                i.name for i in self._interface_outputs
        }

        output_builder_map = {k: _OutputBuilder(k, port_name=self._pythonic_name_to_output_map[k], owner=self)
                              for k in self._pythonic_name_to_output_map.keys()}
        self._outputs: _AttrDict = _AttrDict(output_builder_map)

        self._init_runsettings()
        self._init_k8srunsettings()
        self._init_dynamic_method()
        self._regenerate_output = None

        # add current module to global parent pipeline if there is one
        from .dsl.pipeline import _try_to_add_node_to_current_pipeline
        _try_to_add_node_to_current_pipeline(self)

        # Telemetry
        self._specify_input_mode = False
        self._specify_output_mode = False
        self._specify_output_datastore = False

        _LoggerFactory.trace(_get_logger(), "Module_created", self._get_telemetry_values())

    def set_inputs(self, **kwargs) -> 'Module':
        """Update the inputs of the module."""
        self.inputs.update({k: _InputBuilder(v, k, owner=self) for k, v in kwargs.items() if v is not None})
        return self

    def set_parameters(self, **kwargs) -> 'Module':
        """Update the parameters of the module."""
        self._parameter_params.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    def _init_dynamic_method(self):
        """Update methods set_inputs/set_parameters according to the module input/param definitions."""
        transformed_inputs = self._module_dto.get_transformed_input_params(
            return_yaml=True,
            param_name_dict=self._param_python_name_dict[INPUTS])
        self.set_inputs = create_kw_method_from_parameters(self.set_inputs, transformed_inputs,
                                                           self._old_to_new_param_name_dict)
        transformed_parameters = [
            # Here we set all default values as None to avoid overwriting the values by default values.
            KwParameter(name=param.name, default=None, annotation=param.annotation, _type=param._type)
            for param in self._module_dto.get_transformed_parameter_params(
                return_yaml=True,
                param_name_dict=self._param_python_name_dict[PARAMETERS])
        ]
        self.set_parameters = create_kw_method_from_parameters(self.set_parameters, transformed_parameters,
                                                               self._old_to_new_param_name_dict)

    def _init_runsettings(self):
        run_setting_parameters = self._module_dto.run_setting_parameters
        self._runsettings = _RunSettings(run_setting_parameters, self._module_dto.module_name, self._workspace)

    def _init_k8srunsettings(self):
        self._k8srunsettings = None
        run_setting_parameters = self._module_dto.run_setting_parameters
        target_spec = next((p for p in run_setting_parameters if p.is_compute_target), None)
        mapping = target_spec.run_setting_ui_hint.compute_selection.compute_run_settings_mapping
        compute_settings_types = [compute_type for compute_type in mapping if len(mapping[compute_type]) > 0]
        if len(compute_settings_types) == 0:
            return
        compute_params = mapping[compute_settings_types[0]]
        self._k8srunsettings = _K8sRunSettings(compute_params)
        # Add more docstring to k8srunsettings
        sections_docstring = self._k8srunsettings._doc_string()
        self._k8srunsettings.__doc__ = f"The compute run settings for Module, only take effect " \
                                       f"when compute type is in {compute_settings_types}.\n{sections_docstring}"

    def _build_outputs_map(self, producer=None, default_datastore=None) -> Mapping[str, Any]:
        # output name -> DatasetConsumptionConfig
        _output_map = {}
        for key, val in self._outputs.items():
            if val is None:
                continue
            _output_map[self._pythonic_name_to_output_map[key]] = val.build(producer, default_datastore)

        return _output_map

    def _build_inputs_map(self) -> Mapping[str, Any]:
        # input name -> DatasetConsumptionConfig
        _inputs_map = {}

        for key, val in self._inputs.items():
            if val is None:
                continue
            build = val.build()
            if build is None:
                continue
            _inputs_map[self._pythonic_name_to_input_map[key]] = build

        return _inputs_map

    def _build_params(self) -> Mapping[str, Any]:
        _params = {}

        for key, val in self._parameter_params.items():
            if val is None:
                continue
            if key in self._pythonic_name_to_parameter_map.keys():
                _params[self._pythonic_name_to_parameter_map[key]] = val
        return _params

    def _resolve_compute(self, default_compute, is_local_run=False):
        """
        Resolve compute to tuple

        :param default_compute: pipeline compute specified.
        :type default_compute: tuple(name, type)
        :param is_local_run: whether module execute in local
        :type is_local_run: bool
        :return: (resolve compute, use_module_compute)
        :rtype: tuple(tuple(name, type), bool)
        """
        if not isinstance(default_compute, tuple):
            raise TypeError("default_compute must be a tuple")

        runsettings = self._runsettings
        target = runsettings.target

        if target is None or target == 'local':
            if default_compute[0] is None and not is_local_run:
                raise UserErrorException("A compute target must be specified")
            return default_compute, False

        if isinstance(target, tuple):
            return target, True
        elif isinstance(target, str):
            default_compute_name, _ = default_compute
            if target == default_compute_name:
                return default_compute, True

            # try to resolve
            _targets = self._workspace.compute_targets
            target_in_workspace = _targets.get(target)
            if target_in_workspace is None:
                print('target={}, not found in workspace, assume this is an AmlCompute'.format(target))
                return (target, "AmlCompute"), True
            else:
                return (target_in_workspace.name, target_in_workspace.type), True
        else:
            return target, True

    def _get_telemetry_values(self):
        return self._module_dto.get_telemetry_values(self._workspace)

    def _get_instance_id(self):
        return self._instance_id

    @property
    def name(self):
        """
        Get the name of the Module.

        :return: The name.
        :rtype: str
        """
        return self._module_name

    @property
    def namespace(self):
        """
        Get the namespace of the Module.

        :return: The namespace.
        :rtype: str
        """
        return self._namespace

    @property
    def inputs(self):
        """
        Get the interface inputs of the Module.

        .. remarks::

            You can easily access the inputs interface. The following example shows how it's done.

            .. code-block:: python

                # Suppose ejoin is an `azureml.pipeline.wrapper.Module` instance.
                # Set the input mode.
                ejoin.inputs.left_input.configure(mode='download')

            Note that you should not directly assign things inside `inputs`. For example:

            .. code-block:: python

                # This won't work. Use set_inputs() instead
                ejoin.inputs.left_input = Dataset.get_by_name(ws, 'some dataset')
        """
        return self._inputs

    @property
    def outputs(self):
        """
        Get the interface outputs of the Module.

        .. remarks::

            You can easily access the outputs interface. The following example shows how it's done.

            .. code-block:: python

                # Suppose ejoin is an `azureml.pipeline.wrapper.Module` instance.
                # Set the output datastore.
                ejoin.outputs.ejoin_output.configure(datastore=Datastore(ws, name="myownblob"))

        """
        return self._outputs

    @property
    def runsettings(self):
        """
        The run settings for Module.

        :return: the run settings.
        :rtype: _RunSettings
        """
        return self._runsettings

    @property
    def k8srunsettings(self):
        """The compute run settings for Module
        :return the compute run settings
        :rtype _K8sRunSettings
        """
        return self._k8srunsettings

    @property
    def workspace(self):
        """
        Get the workspace of the Module.

        :return: the Workspace.
        :rtype: azureml.core.Workspace
        """
        return self._workspace

    @property
    def regenerate_output(self):
        """
        The flag whether the module should be run again.

        Set to True to force a new run (disallows module/datasource reuse).

        :return: the regenerate_output value.
        :rtype: bool
        """
        return self._regenerate_output

    @regenerate_output.setter
    def regenerate_output(self, regenerate_output):
        self._regenerate_output = regenerate_output

    def _serialize_inputs_outputs(self):
        real_inputs = [input for input in self.inputs.values() if input.dset is not None]
        inputs_dict = {}
        for input in real_inputs:
            dset = input._get_internal_data_source()
            if isinstance(dset, PipelineParameter):
                if dset.default_value is not None:
                    dset = dset.default_value
                else:
                    # pipeline parameter which has not been assigned
                    dset = dset.name
            if isinstance(dset, DatasetConsumptionConfig):
                dset = dset.dataset

            if isinstance(dset, _OutputBuilder):
                source_val = "{}_{}".format(dset.module_instance_id, dset.port_name)
            elif isinstance(dset, AbstractDataset):
                source_val = dset.name if dset.name is not None else dset.id
            elif isinstance(dset, _GlobalDataset):
                source_val = dset.data_reference_name
            elif isinstance(dset, str):
                source_val = dset
            else:
                raise ValueError("Unknown inputs {}".format(input))

            inputs_dict[self._pythonic_name_to_input_map[input.name]] = {'source': source_val}

        outputs_dict = {}
        for output in self.outputs.values():
            destination = "{}_{}".format(output.module_instance_id, output.port_name)
            outputs_dict[output.port_name] = {'destination': destination}

        return {'inputs': inputs_dict, 'outputs': outputs_dict}

    def _replace(self, new_module):
        """Replace module in pipeline. Use it cautiously"""
        self._module_name = new_module._module_name
        self._namespace = new_module._namespace
        self._module_dto = new_module._module_dto
        self._workspace = new_module._workspace

    def validate(self, fail_fast=False):
        """
        Validate that all the inputs and parameters are in fact valid.

        :param fail_fast: Whether the validation process will fail-fast.
        :type fail_fast: bool

        :return: the errors found during validation.
        :rtype: list
        """
        # Validate inputs
        errors = []

        def process_error(e: Exception, error_type):
            ve = ValidationError(str(e), e, error_type)
            if fail_fast:
                raise ve
            else:
                errors.append({'message': ve.message, 'type': ve.error_type})

        ModuleValidator.validate_module_inputs(provided_inputs=self._inputs,
                                               interface_inputs=self._interface_inputs,
                                               param_python_name_dict=self._param_python_name_dict,
                                               process_error=process_error)

        ModuleValidator.validate_module_parameters(provided_parameters=self._parameter_params,
                                                   interface_parameters=self._interface_parameters,
                                                   param_python_name_dict=self._param_python_name_dict,
                                                   process_error=process_error)

        ModuleValidator.validate_runsettings(runsettings=self._runsettings,
                                             process_error=process_error)

        ModuleValidator.validate_k8srunsettings(k8srunsettings=self._k8srunsettings,
                                                process_error=process_error)

        _LoggerFactory.trace(_get_logger(), "Module_validate", self._get_telemetry_values())
        if len(errors) > 0:
            for error in errors:
                telemetry_value = self._get_telemetry_values()
                telemetry_value.update({
                    'error_message': error['message'],
                    'error_type': error['type']
                })
                _LoggerFactory.trace(_get_logger(), "Module_validate_error", telemetry_value,
                                     adhere_custom_dimensions=False)

        return errors

    def _get_input_config_by_argument_name(self, argument_name):
        inputs_config = self._module_dto.module_entity.structured_interface.inputs
        input_interface = self._module_dto.get_input_interface_by_argument_name(argument_name)
        if input_interface:
            input_name = input_interface.name
        else:
            input_name = self._pythonic_name_to_input_map[argument_name]
        input_config = next(filter(lambda input: input.name == input_name, inputs_config), None)
        return input_config

    def _get_output_config_by_argument_name(self, argument_name):
        outputs_config = self._module_dto.module_entity.structured_interface.outputs
        output_interface = self._module_dto.get_input_interface_by_argument_name(argument_name)
        if output_interface:
            output_name = output_interface.name
        else:
            output_name = self._pythonic_name_to_output_map[argument_name]
        output_config = next(filter(lambda output: output.name == output_name, outputs_config), None)
        return output_config

    @staticmethod
    @track(_get_logger, activity_type=_PUBLIC_API)
    def batch_load(workspace: Workspace, ids: List[str] = None, identifiers: List[tuple] = None) -> \
            List[Callable[..., 'Module']]:
        """
        Batch load modules by identifier list.

        If there is an exception with any module, the batch load will fail. Partial success is not allowed.

        :param workspace: The workspace object this module will belong to.
        :type workspace: azureml.core.Workspace
        :param ids: module version ids
        :type ids: list[str]
        :param identifiers: list of tuple(name, namespace, version)
        :type identifiers: list[azureml.pipeline.wrapper._module.Identifier]

        :return: a tuple of module functions
        :rtype: tuple(function)
        """
        ids, identifiers = _refine_batch_load_input(ids, identifiers, workspace.name)
        service_caller = DesignerServiceCaller(workspace)
        module_number = len(ids) + len(identifiers)
        module_dtos, error_msg = \
            service_caller.batch_get_modules(module_version_ids=ids,
                                             name_identifiers=identifiers)
        if error_msg is not None:
            raise Exception(error_msg)

        module_dtos, failed_id, failed_identifiers = \
            _refine_batch_load_output(module_dtos, ids, identifiers, workspace.name)
        telemetry_values = _get_telemetry_value_from_workspace(workspace)
        telemetry_values.update({
            'count': module_number,
            'failed_id': failed_id,
            'failed_identifiers': failed_identifiers
        })
        _LoggerFactory.trace(_get_logger(), "Module_batch_load", telemetry_values)

        if len(failed_id) > 0 or len(failed_identifiers) > 0:
            raise Exception("Batch load failed, failed module_version_ids: {0}, failed identifiers: {1}".
                            format(failed_id, failed_identifiers))
        module_funcs = (module_dto.to_module_func(workspace,
                                                  module_dto.module_name,
                                                  _ModuleLoadSource.REGISTERED,
                                                  return_yaml=False)
                        for module_dto in module_dtos)
        if module_number == 1:
            module_funcs = next(module_funcs)
        return module_funcs

    @staticmethod
    @track(_get_logger, activity_type=_PUBLIC_API)
    def load(workspace: Workspace, namespace: str = None, name: str = None,
             version: str = None, id: str = None) -> Callable[..., 'Module']:
        """
        Get module function from workspace.

        :param workspace: The workspace object this module will belong to.
        :type workspace: azureml.core.Workspace
        :param namespace: Namespace
        :type namespace: str
        :param name: The name of module
        :type name: str
        :param version: Version
        :type version: str
        :param id: str : The module version id of an existing module
        :type id: str
        :return: a function that can be called with parameters to get a `azureml.pipeline.wrapper.Module`
        :rtype: function
        """
        service_caller = DesignerServiceCaller(workspace)
        if id is None:
            module_dto = service_caller.get_module(
                module_namespace=namespace,
                module_name=name,
                version=version,  # If version is None, this will get the default version
                include_run_setting_params=False
            )
        else:
            module_dto = service_caller.get_module_by_id(module_id=id, include_run_setting_params=False)

        module_dto.correct_module_dto()
        _LoggerFactory.trace(_get_logger(), "Module_load", module_dto.get_telemetry_values(workspace))
        return Module._module_func(workspace, module_dto, module_dto.module_name, _ModuleLoadSource.REGISTERED)

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Module_run')
    def run(self, working_dir=None, experiment_name=None, use_docker=True, track_run_history=True):
        """
        Run module in local container.

        .. remarks::

            After executing module.run, will create scripts, output dirs and log file in working dir.

            .. code-block:: python

                # Suppose we have a workspace as 'ws'
                # First, load a module, and set parameters of module
                ejoin = Module.load(ws, namespace='microsoft.com/bing', name='ejoin')
                module = ejoin(leftcolumns='m:name;age', rightcolumns='income',
                    leftkeys='m:name', rightkeys='m:name', jointype='HashInner')
                # Second, set prepared input path and output path to run module in local. If not set working_dir,
                # will create it in temp dir. In this example, left_input and right_input are input port of ejoin.
                # And after running, output data and log will write in working_dir
                module.set_inputs(left_input=your_prepare_data_path)
                module.set_inputs(right_input=your_prepare_data_path)
                module.run(working_dir=dataset_output_path)

        :param working_dir: The output path for module output info
        :type working_dir: str
        :param experiment_name: The experiment_name will show in portal. If not set, will use module name.
        :type experiment_name: str
        :param use_docker: If use_docker=True, will pull image from azure and run module in container.
                           If use_docker=False, will directly run module script.
        :type use_docker: bool
        :param track_run_history: If track_run_history=True, will create azureml.Run and upload module output
                                  and log file to portal.
                                  If track_run_history=False, will not create azureml.Run to upload outputs
                                  and log file.
        :type track_run_history: bool
        :return: module run status
        :rtype: str
        """
        if not working_dir:
            working_dir = os.path.join(tempfile.gettempdir(), self._module_dto.module_version_id)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        print('working dir is {}'.format(working_dir))

        run = None
        if track_run_history:
            experiment_name = experiment_name if experiment_name else _sanitize_python_variable_name(self._short_name)
            experiment = Experiment(self._workspace, experiment_name)
            run = Run._start_logging(experiment, outputs=working_dir, snapshot_directory=None)
        return _module_run(self, working_dir, use_docker, run=run)

    def _populate_runconfig(self, use_local_compute=False):
        """
        populate runconfig from module dto
        """

        raw_conf = json.loads(self._module_dto.module_entity.runconfig)
        run_config = RunConfiguration._get_runconfig_using_dict(raw_conf)
        run_config._target, compute_type = ('local', None) if use_local_compute else self._runsettings.target

        set_compute_field(self._k8srunsettings, compute_type, run_config)

        if hasattr(self._runsettings, 'process_count_per_node'):
            run_config.mpi.process_count_per_node = self._runsettings.process_count_per_node
        if hasattr(self._runsettings, 'node_count'):
            run_config.node_count = self._runsettings.node_count

        return run_config

    def _transfer_params(self, params, arguments, runconfig):
        """
        transfer params from DataReference to runconfig's setting
        """
        for key, value in params.items():
            k = "--{}".format(key)
            if isinstance(value, DataReference):
                v = "{}{}".format(DATA_REF_PREFIX, key)
                arguments.extend([k, v])
                runconfig.data_references[key] = value.to_config()
            else:
                arguments.extend([k, value])

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='module_submit')
    def submit(self, experiment_name=None, source_dir=None, tags=None) -> Run:
        """Submit module to remote compute target.

        .. remarks::

            Submit is an asynchronous call to the Azure Machine Learning platform to execute a trial on
            remote hardware.  Depending on the configuration, submit will automatically prepare
            your execution environments, execute your code, and capture your source code and results
            into the experiment's run history.
            An example of how to submit an experiment from your local machine is as follows:

            .. code-block:: python

                # Suppose we have a workspace as 'ws'
                # First, load a module, and set parameters of module
                train_module_func = Module.load(ws, namespace='microsoft.com/aml/samples', name='Train')
                train_data = Dataset.get_by_name(ws, 'training_data')
                train = train_module_func(training_data=train_data, max_epochs=5, learning_rate=0.01)
                # Second, set compute target for module then add compute running settings.
                # After running finish, the output data will be in outputs/$output_file
                train.runsettings.configure(target="k80-16-c")
                train.runsettings.resourceconfiguration.configure(gpu_count=1, is_preemptible=True)
                run = train.submit(experiment_name="module-submit-test")
                print(run.get_portal_url())
                run.wait_for_completion()

        :param experiment_name: experiment name
        :type experiment_name: str
        :param source_dir: source dir is where the machine learning scripts locate
        :type source_dir: str
        :param tags: Tags to be added to the submitted run, {"tag": "value"}
        :type tags: dict

        :return run
        :rtype: azureml.core.Run
        """
        if self._runsettings.target is None:
            raise ValueError("module.submit require a remote compute configured.")
        if experiment_name is None:
            experiment_name = _sanitize_python_variable_name(self._short_name)
        if source_dir is None:
            source_dir = os.path.join(tempfile.gettempdir(), self._module_dto.module_version_id)
            print("[Warning] script_dir is None, create tempdir: {}".format(source_dir))
        experiment = Experiment(self._workspace, experiment_name)
        run_config = self._populate_runconfig()

        script = run_config.script
        if not os.path.isfile("{}/{}".format(source_dir, script)):
            print("[Warning] Can't find {} from {}, will download from remote".format(script, source_dir))
            _get_module_snapshot(self, source_dir)

        arguments = []
        default_datastore = Datastore.get(self._workspace, 'workspaceblobstore')
        output = self._build_outputs_map(default_datastore=default_datastore)
        run_id = experiment_name + "_" + str(int(time.time())) + "_" + str(uuid.uuid4())[:8]

        # ScriptRunConfig can't use DataReference directly
        for key, value in output.items():
            k = "--{}".format(key)
            out_key = "outputs"
            v = "{}{}".format(DATA_REF_PREFIX, key)
            arguments.extend([k, v])
            # This path will be showed on portal's outputs if datastore is workspaceblobstore
            path = "{}/{}/{}/{}".format("azureml", run_id, out_key, key)
            run_config.data_references[key] = value.datastore.path(path).to_config()

        all_params = self._build_inputs_map()
        all_params.update(self._parameter_params)
        self._transfer_params(all_params, arguments, run_config)

        src = ScriptRunConfig(source_directory=source_dir, script=script, arguments=arguments, run_config=run_config)
        run = experiment.submit(config=src, tags=tags, run_id=run_id)
        print('Link to Azure Machine Learning Portal:', run.get_portal_url())
        return run

    def _get_arguments(self, input_path, output_path, remove_none_value=True):
        """
        get module arguments

        :param input_path: Replace input port value
        :type input_path: dict
        :param output_path: Replace output port value
        :type output_path: dict
        :return: a list that contains module arguments, and a dict contains environment variables module needed
        :rtype: list, dict
        """

        def _get_argument_value(argument, parameter_params, input_path, output_path):
            arg_value = None

            if argument.value_type == '0':
                # Handle anonymous_module output
                output_prefix = 'DatasetOutputConfig:'
                if argument.value.startswith(output_prefix):
                    arg_value_key = _get_or_sanitize_python_name(argument.value[len(output_prefix):],
                                                                 self._param_python_name_dict[OUTPUTS])
                    if arg_value_key in output_path:
                        return output_path[arg_value_key]
                return argument.value
            elif argument.value_type == '1':
                # Get parameter argument value
                arg_value_key = \
                    _get_or_sanitize_python_name(argument.value, self._param_python_name_dict[PARAMETERS])
                if arg_value_key in parameter_params.keys():
                    if parameter_params[arg_value_key] is not None:
                        arg_value = str(parameter_params[arg_value_key])
                    else:
                        arg_value = parameter_params[arg_value_key]
                else:
                    arg_value = argument.value
            elif argument.value_type == '2':
                # Get input argument value
                arg_value_key = \
                    _get_or_sanitize_python_name(argument.value, self._param_python_name_dict[INPUTS])
                if arg_value_key in input_path.keys():
                    arg_value = input_path[arg_value_key]
                else:
                    arg_value = None
            elif argument.value_type == '3':
                # Get output argument value
                arg_value_key = \
                    _get_or_sanitize_python_name(argument.value, self._param_python_name_dict[OUTPUTS])
                if arg_value_key in output_path:
                    arg_value = output_path[arg_value_key]
            elif argument.value_type == '4':
                # Get nestedList argument value
                arg_value = []
                for sub_arg in argument.nested_argument_list:
                    sub_arg_value = _get_argument_value(
                        sub_arg, parameter_params, input_path, output_path)
                    arg_value.append(sub_arg_value)
            return arg_value

        arguments = self._module_dto.module_entity.structured_interface.arguments
        parameter_params = self._parameter_params
        ret = []
        runconfig = json.loads(self._module_dto.module_entity.runconfig)
        framework = None if 'Framework' not in runconfig else runconfig['Framework']
        if framework and framework.lower() == 'python':
            ret.append('python')
        elif not framework:
            raise Exception('Framework in runconfig is None')
        else:
            raise NotImplementedError(f'Unsupported framework {framework}, only Python is supported now.')
        script = None if 'Script' not in runconfig else runconfig['Script']
        if script:
            ret.append(script)
        else:
            raise Exception('Script in runconfig is None')
        for arg in arguments:
            arg_value = _get_argument_value(arg, parameter_params, input_path, output_path)
            ret.append(arg_value)

        def flatten(arr):
            for element in arr:
                if hasattr(element, "__iter__") and not isinstance(element, str):
                    for sub in flatten(element):
                        yield sub
                else:
                    yield element

        ret = list(flatten(ret))
        if remove_none_value:
            # Remove None value and its flag in arguments
            ret = [x[0] for x in zip(ret, ret[1:] + ret[-1::]) if (x[0] and x[1])]
        return ret

    @staticmethod
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_func(workspace: Workspace, func: types.FunctionType, force_reload=True) -> Callable[..., 'Module']:
        """
        Register an anonymous module from a wrapped python function. Then return the registered module func.

        :param workspace: The workspace object this module will belong to.
        :type workspace: azureml.core.Workspace
        :param func: A wrapped function to be loaded or a ModuleExecutor instance.
        :type func: types.FunctionType
        :param force_reload: Whether reload the function to make sure the code is the latest.
        :type force_reload: bool
        """
        def _reload_func(f: types.FunctionType):
            """Reload the function to make sure the latest code is used to generate yaml."""
            module = importlib.import_module(f.__module__)
            if module.__spec__ is not None:
                # Reload the module except the case that module.__spec__ is None.
                # In the case module.__spec__ is None (E.g. module is __main__), reload will raise exception.
                importlib.reload(module)
            return getattr(module, f.__name__)

        if force_reload:
            func = _reload_func(func)
        # Import here to avoid circular import.
        from azureml.pipeline.wrapper.dsl.module import ModuleExecutor
        from azureml.pipeline.wrapper.dsl._module_spec import SPEC_EXT
        # If a ModuleExecutor instance is passed, we directly use it,
        # otherwise we construct a ModuleExecutor with the function
        executor = func if isinstance(func, ModuleExecutor) else ModuleExecutor(func)
        # Use a temp spec file to register.
        temp_spec_file = '_' + Path(inspect.getfile(func)).with_suffix(SPEC_EXT).name
        temp_spec_file = executor.to_spec_yaml(spec_file=temp_spec_file)
        try:
            _LoggerFactory.trace(_get_logger(), "Module_from_func",
                                 _get_telemetry_value_from_workspace(workspace))
            return Module.from_yaml(workspace=workspace, yaml_file=str(temp_spec_file), _from_func=True)
        finally:
            if temp_spec_file.is_file():
                temp_spec_file.unlink()

    @staticmethod
    @track(_get_logger, activity_type=_PUBLIC_API)
    def from_yaml(workspace: Workspace, yaml_file: str, _from_func=False) -> Callable[..., 'Module']:
        """
        Register an anonymous module from yaml file to workspace.

        Assumes source code is in the same directory with yaml file. Then return the registered module func.

        :param workspace: The workspace object this module will belong to.
        :type workspace: azureml.core.Workspace
        :param yaml_file: Module spec file. The spec file could be located in local or Github.
                          For example:

                          * "custom_module/module_spec.yaml"
                          * "https://github.com/zzn2/sample_modules/blob/master/3_basic_module/basic_module.yaml"
        :type yaml_file: str
        :return: a function that can be called with parameters to get a `azureml.pipeline.wrapper.Module`
        :rtype: function
        """
        module_dto = _load_anonymous_module(workspace, yaml_file)

        # build module func with module version
        module_dto.correct_module_dto()
        _LoggerFactory.trace(_get_logger(), "Module_from_yaml", module_dto.get_telemetry_values(workspace))
        return Module._module_func(workspace, module_dto, module_dto.module_name,
                                   _ModuleLoadSource.FROM_FUNC if _from_func else _ModuleLoadSource.FROM_YAML)

    @staticmethod
    @track(_get_logger, activity_type=_PUBLIC_API)
    def register(workspace: Workspace, yaml_file: str, amlignore_file: str = None, set_as_default: bool = False) -> \
            Callable[..., 'Module']:
        """
        Register an module from yaml file to workspace.

        Assumes source code is in the same directory with yaml file. Then return the registered module func.

        :param workspace: The workspace object this module will belong to.
        :type workspace: azureml.core.Workspace
        :param yaml_file: Module spec file. The spec file could be located in local or Github.
                          For example:

                          * "custom_module/module_spec.yaml"
                          * "https://github.com/zzn2/sample_modules/blob/master/3_basic_module/basic_module.yaml"
        :type yaml_file: str
        :param amlignore_file: The .amlignore or .gitignore file path used to exclude files/directories in the snapshot
        :type amlignore_file: str
        :param set_as_default: By default false, default version of the module will not be updated
                                when registering a new version of module. Specify this flag to set
                                the new version as the module's default version.
        :type set_as_default: bool
        :return: a function that can be called with parameters to get a `azureml.pipeline.wrapper.Module`
        :rtype: function
        """

        module_dto = _register_module_from_yaml(workspace, yaml_file, amlignore_file=amlignore_file,
                                                set_as_default=set_as_default)

        # build module func with module dto
        _LoggerFactory.trace(_get_logger(), "Module_register", module_dto.get_telemetry_values(workspace))
        return Module._module_func(workspace, module_dto, module_dto.module_name, _ModuleLoadSource.FROM_YAML)

    @staticmethod
    def _module_func(workspace: Workspace, module_dto: ModuleDto, name: str,
                     _load_source: str = _ModuleLoadSource.UNKNOWN) -> Callable[..., 'Module']:
        """
        Get module func from ModuleDto

        :param workspace: The workspace object this module will belong to.
        :type workspace: azureml.core.Workspace
        :param module_dto: ModuleDto instance
        :type module_dto: azureml.pipeline.wrapper._module_dto
        :param name: The name of module
        :type name: str
        :param _load_source: The source which the module is loaded.
        :type _load_source: str
        :return: a function that can be called with parameters to get a `azureml.pipeline.wrapper.Module`
        :rtype: function
        """
        return module_dto.to_module_func(workspace, name, _load_source)

    def _is_target_module(self, target):
        """
        provided for replace a module in pipeline
        check if current node(module) is the target one we want

        :return: Result of comparision between two modules
        :rtype: bool
        """
        if target.name != self.name:
            return False
        if target._namespace != self._namespace:
            return False
        if target._module_dto != self._module_dto:
            return False
        return True
