# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from typing import Callable, List
from datetime import timezone
from azureml.core import Workspace
from ._restclients.designer.models import ModuleDto as DesignerModuleDto, RunSettingParameterType, \
    RunSettingUIWidgetTypeEnum
from ._dynamic import KwParameter, create_kw_function_from_parameters
from ._utils import _sanitize_python_variable_name, _get_or_sanitize_python_name
from ._telemetry import _get_telemetry_value_from_workspace, _get_telemetry_value_from_module_dto

PARAMETERS = 'parameters'
OUTPUTS = 'outputs'
INPUTS = 'inputs'
IGNORE_PARAMS = {
    # FixedParams
    'ServingEntry', 'Target', 'MLCComputeType', 'PrepareEnvironment',
    'Script', 'Framework', 'MaxRunDurationSeconds', 'InterpreterPath',
    'UserManagedDependencies', 'CondaDependencies', 'DockerEnabled', 'BaseDockerImage',
    'GpuSupport', 'HistoryEnabled', 'Arguments',
    # MPI, TODO confirm
    'Communicator', 'MpiProcessCountPerNode', 'NodeCount',
}


def _get_docstring_lines(all_params, all_outputs, param_name_dict):
    docstring_lines = []
    if len(all_params) > 0:
        docstring_lines.append("\n")
    for param in all_params:
        if param.default is None or (isinstance(param.default, str) and len(param.default.strip()) == 0):
            docstring_lines.append(":param {}: {}".format(param.name, param.annotation))
        else:
            docstring_lines.append(":param {}: {}. (optional, default value is {}.)"
                                   .format(param.name, param.annotation, param.default))
        docstring_lines.append(":type {}: {}".format(param.name, param._type))

    if len(all_outputs) > 0:
        docstring_lines.append("\n")
    for o in all_outputs:
        output_name = \
            _get_or_sanitize_python_name(o.name, param_name_dict[OUTPUTS])
        docstring_lines.append(":output {}: {}".format(output_name,
                                                       o.description if o.description is not None else o.name))
        docstring_lines.append(":type: {}: {}".format(
            output_name, str(o.data_type_id)))
    return docstring_lines


def _type_code_to_python_type(type_code):
    type_code = int(type_code)
    if type_code == 0:
        return int
    elif type_code == 1:
        return float
    elif type_code == 2:
        return bool
    elif type_code == 3:
        return str


def _type_code_to_python_type_name(type_code):
    try:
        return _type_code_to_python_type(type_code).__name__
    except BaseException:
        return None


def _python_type_to_type_code(value):
    if value is int:
        return 0
    if value is float:
        return 1
    if value is bool:
        return 2
    if value is str:
        return 3


def _int_str_to_run_setting_parameter_type(int_str_value):
    return list(RunSettingParameterType)[int(int_str_value)]


def _int_str_to_run_setting_ui_widget_type_enum(int_str_value):
    return list(RunSettingUIWidgetTypeEnum)[int(int_str_value)]


def _run_setting_param_type_to_python_type(param_type: RunSettingParameterType):
    if param_type == RunSettingParameterType.json_string or param_type == RunSettingParameterType.string:
        return str
    if param_type == RunSettingParameterType.double:
        return float
    if param_type == RunSettingParameterType.int_enum:
        return int
    if param_type == RunSettingParameterType.bool_enum:
        return bool


class ModuleDto(DesignerModuleDto):
    def __init__(self, module_dto: DesignerModuleDto = None, **kwargs):
        if module_dto:
            _dict = {k: v for k, v in module_dto.__dict__.items()
                     if k in DesignerModuleDto._attribute_map.keys()}
            kwargs.update(_dict)
        super().__init__(**kwargs)
        self.correct_run_settings()

    def get_telemetry_values(self, workspace: Workspace):
        telemetry_values = {}
        telemetry_values.update(_get_telemetry_value_from_workspace(workspace))
        telemetry_values.update(_get_telemetry_value_from_module_dto(self))
        return telemetry_values

    def get_module_param_dict_list(self, _type: str):
        """
        Get dict list of parameters with argumentName and
        commandLineOption inside it, example result:
        [{'name': 'Output_path', 'argumentName': 'output_mark',
        'commandLineOption': '--output-path'}]

        :param self: the module dto
        :type self: ModuleDto
        :param _type: parameter type string,
            possible value in ('inputs', 'outputs', 'parameters')
        :type _type: str
        :return: PythonInterfaceMapping list
        :rtype: list[~designer.models.PythonInterfaceMapping]
        """
        if self.module_python_interface is None:
            return []

        if _type is INPUTS:
            return self.module_python_interface.inputs
        elif _type is OUTPUTS:
            return self.module_python_interface.outputs
        elif _type is PARAMETERS:
            return self.module_python_interface.parameters
        else:
            raise Exception('Unknown parameter type "{0}"'.format(_type))

    def get_module_param_python_name_dict(self):
        param_dict = {INPUTS: {param.name: param.argument_name for param in
                               self.get_module_param_dict_list(INPUTS)},
                      OUTPUTS: {param.name: param.argument_name for param in
                                self.get_module_param_dict_list(OUTPUTS)},
                      PARAMETERS: {param.name: param.argument_name for param in
                                   self.get_module_param_dict_list(PARAMETERS)}}
        return param_dict

    def get_old_to_new_param_name_dict(self):
        return {_type: {_sanitize_python_variable_name(k): v
                        for k, v in _items.items()}
                for _type, _items in
                self.get_module_param_python_name_dict().items()}

    def get_refined_module_dto_identifiers(self, workspace_name):
        identifiers = [self.module_name, (self.module_name, self.namespace),
                       (self.module_name, self.namespace, self.module_version)]
        from ._module import _refine_batch_load_input
        _, identifiers = _refine_batch_load_input([], identifiers, workspace_name)
        return identifiers

    def to_module_func(self, ws: Workspace, module_name: str, _load_source: str, return_yaml: bool = True) -> Callable:
        func_docstring_lines = []
        module_description = self.description.strip() if self.description else ""
        func_docstring_lines.append(module_description)

        interface: StructuredInterface = self.module_entity.structured_interface
        param_name_dict = self.get_module_param_python_name_dict()
        transformed_inputs = \
            self.get_transformed_input_params(return_yaml,
                                              param_name_dict[INPUTS])
        transformed_parameters = \
            self.get_transformed_parameter_params(return_yaml,
                                                  param_name_dict[PARAMETERS])

        all_params = transformed_inputs + transformed_parameters
        all_outputs = interface.outputs

        from ._module import Module

        def create_module_func_from_model_version(**kwargs) -> Module:
            return Module(ws, self, kwargs, module_name, _load_source)

        namespace = self.namespace
        if namespace is None:
            func_name = "[module] {}".format(module_name)
        else:
            func_name = "[module] {} (namespace: {})".format(module_name, namespace)

        from ._module import _get_module_yaml
        yaml = \
            _get_module_yaml(ws, self.module_version_id) if return_yaml else None
        if yaml:
            # Use ```yaml\n{1}\n``` to make the yaml part readable in markdown.
            doc_string = '{0}\n\nModule yaml:\n```yaml\n{1}\n```'.format(module_description, yaml)
        else:
            doc_string = '{0}\n\n{1}'.format(
                module_description,
                '\n'.join(_get_docstring_lines(all_params, all_outputs, param_name_dict)))
        dynamic_func = create_kw_function_from_parameters(
            create_module_func_from_model_version,
            documentation=doc_string,
            parameters=all_params,
            func_name=func_name,
            old_to_new_param_name_dict=self.get_old_to_new_param_name_dict()
        )

        return dynamic_func

    def correct_module_dto(self):
        if not isinstance(self, ModuleDto):
            return self
        # A module_dto may not have a valid created_date so tzinfo is not set.
        # In such case we manually set the tzinfo to avoid the following warning.
        # WARNING - Datetime with no tzinfo will be considered UTC.
        # This warning is printed when serializing to json in the following code.
        # https://github.com/Azure/msrest-for-python/blob/master/msrest/serialization.py#L1039
        if self.created_date.tzinfo is None:
            self.created_date = self.created_date.replace(tzinfo=timezone.utc)
        if self.last_modified_date.tzinfo is None:
            self.last_modified_date = self.last_modified_date.replace(tzinfo=timezone.utc)

        module_entity = self.module_entity
        if module_entity.created_date.tzinfo is None:
            module_entity.created_date = module_entity.created_date.replace(tzinfo=timezone.utc)
        if module_entity.last_modified_date.tzinfo is None:
            module_entity.last_modified_date = module_entity.last_modified_date.replace(tzinfo=timezone.utc)

    def _correct_run_settings_param(self, param):
        # Convert int string to enum type
        param.parameter_type = _int_str_to_run_setting_parameter_type(param.parameter_type)
        param.parameter_type_in_py = _run_setting_param_type_to_python_type(param.parameter_type)
        param.run_setting_ui_hint.ui_widget_type = _int_str_to_run_setting_ui_widget_type_enum(
            param.run_setting_ui_hint.ui_widget_type)
        # Convert default value to correct type
        if param.default_value is not None and param.parameter_type_in_py is not None:
            param.default_value = param.parameter_type_in_py(param.default_value)
        # Handle none argument name
        if not hasattr(param, 'argument_name') or param.argument_name is None:
            param.argument_name = _sanitize_python_variable_name(param.label)

    def correct_run_settings(self):
        run_setting_parameters = self.run_setting_parameters
        for p in run_setting_parameters:
            self._correct_run_settings_param(p)
            # Mark compute target param
            p.is_compute_target = p.run_setting_ui_hint.ui_widget_type == RunSettingUIWidgetTypeEnum.compute_selection
        # Compute run settings
        target_param = next((p for p in run_setting_parameters if p.is_compute_target), None)
        compute_specs = target_param.run_setting_ui_hint.compute_selection.compute_run_settings_mapping
        for compute_type in compute_specs:
            compute_params = compute_specs[compute_type]
            if len(compute_params) > 0:
                for p in compute_params:
                    if not hasattr(p, 'section_argument_name'):
                        p.section_argument_name = _sanitize_python_variable_name(p.section_name)
                    self._correct_run_settings_param(p)

    def get_transformed_input_params(self, return_yaml: bool,
                                     param_name_dict: dict) -> List[KwParameter]:
        inputs = self.module_entity.structured_interface.inputs
        return [
            KwParameter(
                name=_get_or_sanitize_python_name(i.name, param_name_dict),
                # annotation will be parameter type if return yaml
                annotation=i.label if not return_yaml
                else str(i.data_type_ids_list),
                default=None,
                _type=str(i.data_type_ids_list)
            )
            for i in inputs
        ]

    def get_transformed_parameter_params(self, return_yaml: bool,
                                         param_name_dict: dict) -> List[KwParameter]:
        parameters = self.module_entity.structured_interface.parameters
        return [
            KwParameter(
                name=_get_or_sanitize_python_name(p.name, param_name_dict),
                # annotation will be parameter type if return yaml
                annotation=p.label if not return_yaml
                else _type_code_to_python_type_name(p.parameter_type),
                default=p.default_value,
                _type=_type_code_to_python_type_name(p.parameter_type)
            )
            for p in parameters if p.name not in IGNORE_PARAMS
        ]

    def get_input_interface_by_argument_name(self, argument_name):
        inputs_interface = self.get_module_param_dict_list(INPUTS)
        input_interface = next(
            filter(lambda input: input.argument_name == argument_name, inputs_interface), None)
        return input_interface
