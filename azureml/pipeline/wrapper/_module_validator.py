# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from typing import List
import warnings

from azureml.exceptions._azureml_exception import UserErrorException

from ._restclients.designer.models import StructuredInterfaceParameter, RunSettingParameterType
from ._utils import _get_or_sanitize_python_name, is_float_convertible, is_bool_string, is_int_convertible, \
    _is_json_string_convertible
from ._module_dto import _type_code_to_python_type_name, \
    IGNORE_PARAMS, INPUTS, PARAMETERS
from ._pipeline_parameters import PipelineParameter


class ValidationError(UserErrorException):
    """
    Exception raised when module validation fails.
    """

    UNKNOWN = 'Unknown'
    MISSING_INPUT = 'MissingInput'
    MISSING_PARAMETER = 'MissingParameter'
    MISSING_RUNSETTING_PARAMETER = 'MissingRunSettingParameter'
    PARAMETER_TYPE_MISMATCH = 'ParameterTypeMismatch'
    INVALID_PARAMETER = 'InvalidParameter'
    INVALID_RUNSETTING_PARAMETER = 'InvalidRunSettingParameter'

    def __init__(self, message, inner_exception=None, error_type=UNKNOWN):
        super().__init__(message)
        self.inner_exception = inner_exception
        self.error_type = error_type


class ModuleValidator(object):

    @staticmethod
    def validate_module_inputs(provided_inputs, interface_inputs, param_python_name_dict, process_error):
        for _input in interface_inputs:
            formatted_input_name = \
                _get_or_sanitize_python_name(_input.name, param_python_name_dict[INPUTS])
            provided_input_value = provided_inputs.get(formatted_input_name, None).build()
            if not _input.is_optional and provided_input_value is None:
                process_error(ValueError(f"Required input {formatted_input_name} not provided."),
                              ValidationError.MISSING_INPUT)
                continue
            # TODO: input port type validation?

    @staticmethod
    def validate_module_parameters(provided_parameters, interface_parameters, param_python_name_dict, process_error):

        def type_mismatch(parameter_name, _type, param_value):
            process_error(ValueError(f"Parameter {parameter_name} type mismatch. "
                                     f"Required {_type}, got {type(param_value)}."),
                          ValidationError.PARAMETER_TYPE_MISMATCH)

        def validate_numeric_parameter(parameter_name, _type, param_value):
            # Try to convert string parameter to int/float
            if isinstance(param_value, str) and parameter_type == '0' \
                    and is_int_convertible(param_value):
                param_value = int(param_value)
            if isinstance(param_value, str) and parameter_type == '1' \
                    and is_float_convertible(param_value):
                param_value = float(param_value)
            if isinstance(param_value, bool):
                type_mismatch(parameter_name, _type, param_value)
                return
            # Don't allow other types when int is required
            if parameter_type == '0' and not isinstance(param_value, int):
                type_mismatch(parameter_name, _type, param_value)
                return
            # Allow int and float when float is required
            if parameter_type == '1' and \
                    not (isinstance(param_value, int) or isinstance(param_value, float)):
                type_mismatch(parameter_name, _type, param_value)
                return

            lower_bound = parameter.lower_bound
            if lower_bound is not None and param_value < float(lower_bound):
                process_error(ValueError(f"Parameter {parameter_name} range is invalid. "
                                         f"Lower bound is {lower_bound}, got {param_value}."),
                              ValidationError.INVALID_PARAMETER)
            upper_bound = parameter.upper_bound
            if upper_bound is not None and param_value > float(upper_bound):
                process_error(ValueError(f"Parameter {parameter_name} range is invalid. "
                                         f"Upper bound is {upper_bound}, got {param_value}."),
                              ValidationError.INVALID_PARAMETER)

        def validate_enum_values_parameter(parameter_name, _type, param_value):
            if not isinstance(param_value, str):
                type_mismatch(parameter_name, _type, param_value)
                return
            enum_values = parameter.enum_values
            if enum_values is not None and param_value not in enum_values:
                process_error(ValueError(f"Parameter {parameter_name} is invalid. "
                                         f"Options are {str(enum_values)}, got '{param_value}'."),
                              ValidationError.INVALID_PARAMETER)

        def _is_conditional_optional(parameter, provided_parameters):
            # this is to support build-in modules' conditional required parameter
            # e.g. 'Split Data' has parameter named 'Stratification Key Column' which is configured
            # to be enabled by parameter 'Stratified split' with value set ['True']
            # so if parameter 'Stratified split' is provided with value equals to 'True'
            # we should consider 'Stratification Key Column' as a required parameter, otherwise as optional
            if parameter.enabled_by_parameter_name is None or parameter.enabled_by_parameter_values is None:
                return parameter.is_optional

            enabled_by = parameter.enabled_by_parameter_name
            enabled_by_values = parameter.enabled_by_parameter_values
            provided_value = provided_parameters.get(
                _get_or_sanitize_python_name(enabled_by, param_python_name_dict[PARAMETERS]), None)
            return provided_value not in enabled_by_values

        # Validate params
        for parameter in interface_parameters:
            if parameter.name in IGNORE_PARAMS:
                continue

            formatted_parameter_name = \
                _get_or_sanitize_python_name(parameter.name, param_python_name_dict[PARAMETERS])
            provided_param_value = provided_parameters.get(formatted_parameter_name, None)
            is_parameter_optional = _is_conditional_optional(parameter, provided_parameters)
            if provided_param_value is None:
                if not is_parameter_optional:
                    process_error(ValueError(f"Required parameter {formatted_parameter_name} not provided."),
                                  ValidationError.MISSING_PARAMETER)
                continue
            else:
                parameter_type = parameter.parameter_type
                if isinstance(provided_param_value, PipelineParameter):
                    provided_param_value = provided_param_value.default_value

                required_parameter_type = _type_code_to_python_type_name(parameter_type)

                # '0' means int type, '1' means float type:
                if parameter_type == '0' or parameter_type == '1':
                    validate_numeric_parameter(formatted_parameter_name, required_parameter_type, provided_param_value)
                # '2' means boolean
                elif parameter_type == '2':
                    if not (isinstance(provided_param_value, bool) or is_bool_string(provided_param_value)):
                        type_mismatch(formatted_parameter_name, required_parameter_type, provided_param_value)
                # '3' means str type
                elif parameter_type == '3':
                    validate_enum_values_parameter(formatted_parameter_name, required_parameter_type,
                                                   provided_param_value)

    @staticmethod
    def validate_compatibility(old_module, new_module):
        """
        provided for replace a module in pipeline
        compare ports and params
        """
        errors = []
        ModuleValidator._validate_ports(old_module, new_module, errors)
        ModuleValidator._validate_parameters(old_module._interface_parameters, new_module._interface_parameters,
                                             errors)
        return errors

    @staticmethod
    def _validate_ports(old_module, new_module, errors: List):
        """
        validate input and output ports defined in module
        both name and mode, allow additional
        """

        def _input_provided(module, port_name: str):
            provided_inputs = module._inputs
            formatted_input_name = _get_or_sanitize_python_name(port_name, module._param_python_name_dict[INPUTS])
            provided_input_value = provided_inputs.get(formatted_input_name, None)
            return provided_input_value is not None

        def _check_missing(ptype: str, old_ports: dict, new_ports: dict, errors: List):
            missing_ports = list(old_ports.keys() - new_ports.keys())
            if len(missing_ports) != 0:
                errors.append("Missing {0} ports in new module function, expected {1}, but not found.".
                              format(ptype, missing_ports))
            # only check inputs now
            if ptype == 'Output':
                return
            mismatched_ports = [[k, v.data_type_ids_list, new_ports[k].data_type_ids_list]
                                for k, v in old_ports.items() if k in new_ports.keys() and
                                _input_provided(old_module, k) and
                                len(set(v.data_type_ids_list) - set(new_ports[k].data_type_ids_list)) > 0]
            # flatten mismatched ports
            mismatched_ports = [mismatched_ports[i][j] for i in range(len(mismatched_ports))
                                for j in range(len(mismatched_ports[i]))]
            if len(mismatched_ports) != 0:
                errors.append("{0} ports data type mismatched {1}, excepted type: {2}, actually {3}.".
                              format(ptype, mismatched_ports[0::3], mismatched_ports[1::3],
                                     mismatched_ports[2::3]))
            # check if required port added
            required_ports = {p.name: p.data_type_ids_list for p in new_ports.values()
                              if p.name not in old_ports.keys() and not p.is_optional}
            if len(required_ports) != 0:
                errors.append("New required ports {0} added in new module function, type {1}".
                              format(required_ports.keys(), required_ports.values()))

        old_inputs = {i.name: i for i in old_module._interface_inputs}
        new_inputs = {i.name: i for i in new_module._interface_inputs}
        _check_missing("Input", old_inputs, new_inputs, errors)

        old_outputs = {i.name: i for i in old_module._interface_outputs}
        new_outputs = {i.name: i for i in new_module._interface_outputs}
        _check_missing("Output", old_outputs, new_outputs, errors)
        return errors

    @staticmethod
    def _validate_parameters(old_params, new_params: List[StructuredInterfaceParameter],
                             errors: List):
        """
        validate parameters defined in module with new module's params
        type of interface parameter definition:  List[StructuredInterfaceParameter]

        only compare name and type now, allow additional params
        """
        old_param_dict = {p.name: p.parameter_type for p in old_params}
        new_param_dict = {p.name: p.parameter_type for p in new_params}

        if old_param_dict == new_param_dict:
            return
        # missing params: new function does not contains parameters in the old one
        # check at first to show more clear error message
        missing_params = {p.name: _type_code_to_python_type_name(p.parameter_type)
                          for p in old_params if p.name not in new_param_dict.keys()}
        if len(missing_params) != 0:
            errors.append("Missing parameter in new module function, excepted: {0}, but not found.".
                          format(missing_params))

        # mismatched params: new function has some parameters that type mismatched with the old one
        # looks like this: [['name', 'str', 'int'][..]..]
        # the 2nd is type excepted and the 3rd is type real
        mismatched_params = [[k, _type_code_to_python_type_name(v),
                              _type_code_to_python_type_name(new_param_dict[k])]
                             for k, v in old_param_dict.items() if k in new_param_dict.keys() and
                             v != new_param_dict[k]]
        # flatten mismatched params
        mismatched_params = [mismatched_params[i][j] for i in range(len(mismatched_params))
                             for j in range(len(mismatched_params[i]))]
        if len(mismatched_params) != 0:
            errors.append("Parameter type mismatched {0}, excepted type: {1}, actually {2}.".
                          format(mismatched_params[0::3], mismatched_params[1::3],
                                 mismatched_params[2::3]))
        # check if new module includes required params which not exists in old module
        required_params = {p.name: _type_code_to_python_type_name(p.parameter_type) for p in new_params
                           if p.name not in old_param_dict.keys() and not p.is_optional}
        if len(required_params) != 0:
            errors.append("No such required params {0} provided in old module, type: {1}.".
                          format(required_params.keys(), required_params.values()))

    @staticmethod
    def validate_runsettings(runsettings, process_error):
        for param_name in runsettings._params_spec:
            param_spec = runsettings._params_spec[param_name]
            if param_spec.is_compute_target:
                # Bypass target validation since we have done it in property setter
                continue
            param_value = getattr(runsettings, param_name, None)
            ModuleValidator._validate_runsetting_parameter(f"runsettings.{param_name}",
                                                           param_value,
                                                           param_spec,
                                                           process_error)
        # Raise warnings for parameters not in spec
        reserved_param_names = [
            '_target', 'configure', '_workspace', '_params_spec', '__doc__']
        ModuleValidator._raise_warnings_for_params_not_in_spec(reserved_param_names,
                                                               runsettings._params_spec.keys(),
                                                               vars(runsettings).keys(),
                                                               'runsettings')

    @staticmethod
    def validate_single_runsettings_parameter(param_name, param_value, runsettings, process_error):
        if param_name == "target":
            return  # Bypass target validation since we have done it in property setter
        spec = runsettings._params_spec[param_name]
        ModuleValidator._validate_runsetting_parameter(f"runsettings.{param_name}",
                                                       param_value,
                                                       spec,
                                                       process_error)

    @staticmethod
    def validate_k8srunsettings(k8srunsettings, process_error):
        if k8srunsettings is None:
            return
        for section_name in k8srunsettings._params_spec:
            section_settings = getattr(k8srunsettings, section_name, None)
            if section_settings is None:
                continue  # k8srunsettings is all optional
            for param_name in section_settings._params_spec:
                param_spec = section_settings._params_spec[param_name]
                param_value = getattr(section_settings, param_name, None)
                ModuleValidator._validate_runsetting_parameter(f"k8srunsettings.{param_name}",
                                                               param_value,
                                                               param_spec,
                                                               process_error)
            # Raise warnings for parameters not in spec
            reserved_param_names = ['_params_spec', '_section_name', '__doc__', 'configure']
            ModuleValidator._raise_warnings_for_params_not_in_spec(reserved_param_names,
                                                                   section_settings._params_spec.keys(),
                                                                   vars(section_settings).keys(),
                                                                   f"k8srunsettings.{section_name}")

        # Raise warnings for sections not in spec
        reserved_section_names = ['_params_spec', '__doc__']
        ModuleValidator._raise_warnings_for_params_not_in_spec(reserved_section_names,
                                                               k8srunsettings._params_spec.keys(),
                                                               vars(k8srunsettings).keys(),
                                                               'k8srunsettings')

    @staticmethod
    def validate_single_k8srunsettings_parameter(param_name, param_value, section_settings, process_error):
        spec = section_settings._params_spec[param_name]
        ModuleValidator._validate_runsetting_parameter(f"k8srunsettings.{section_settings._section_name}.{param_name}",
                                                       param_value,
                                                       spec,
                                                       process_error)

    @staticmethod
    def _get_runsetting_advanced_validator_by_type(parameter_type):
        if parameter_type == RunSettingParameterType.int_enum or parameter_type == RunSettingParameterType.double:
            return ModuleValidator._validate_numeric_runsetting_parameter
        if parameter_type == RunSettingParameterType.json_string:
            return ModuleValidator._validate_json_string_runsetting_parameter
        return None

    @staticmethod
    def _validate_runsetting_parameter(param_hint_name, param_value, spec, process_error):
        if param_value is None:
            # Required parameter validation
            if spec.is_optional is False:
                process_error(ValueError(f"Required parameter '{param_hint_name}' not provided."),
                              ValidationError.MISSING_RUNSETTING_PARAMETER)
            return
        # Type validation
        if not isinstance(param_value, spec.parameter_type_in_py):
            # Try to convert
            try:
                param_value = spec.parameter_type_in_py(param_value)
            except Exception:
                process_error(ValueError(f"Parameter '{param_hint_name}' is invalid, which cannot be convert to"
                                         f" expected type '{spec.parameter_type_in_py.__name__}'"),
                              ValidationError.INVALID_RUNSETTING_PARAMETER)
        # Value validation
        advanced_validator = ModuleValidator._get_runsetting_advanced_validator_by_type(spec.parameter_type)
        if advanced_validator is None:
            return  # Pass
        advanced_validator(param_hint_name, param_value, spec, process_error)

    @staticmethod
    def _validate_numeric_runsetting_parameter(param_hint_name, param_value, spec, process_error):
        upper_bound = spec.parameter_type_in_py(spec.upper_bound) if spec.upper_bound is not None else None
        lower_bound = spec.parameter_type_in_py(spec.lower_bound) if spec.lower_bound is not None else None
        if upper_bound is not None and param_value > upper_bound:
            process_error(ValueError(f"Parameter '{param_hint_name}' is invalid, which should '<= {upper_bound}', "
                                     f"got '{param_value}'"),
                          ValidationError.INVALID_RUNSETTING_PARAMETER)
        if lower_bound is not None and param_value < lower_bound:
            process_error(ValueError(f"Parameter '{param_hint_name}' is invalid, which should '>= {lower_bound}', "
                                     f"got '{param_value}'"),
                          ValidationError.INVALID_RUNSETTING_PARAMETER)

    @staticmethod
    def _validate_json_string_runsetting_parameter(param_hint_name, param_value, spec, process_error):
        if not _is_json_string_convertible(param_value):
            process_error(ValueError(f"Parameter '{param_hint_name}' is invalid, "
                                     "which should be a valid JSON string"),
                          ValidationError.INVALID_RUNSETTING_PARAMETER)

    @staticmethod
    def _raise_warnings_for_params_not_in_spec(reserved_param_names,
                                               param_names_in_spec,
                                               provided_param_names,
                                               config_name):
        expected_param_names = set(reserved_param_names).union(set(param_names_in_spec))
        params_not_in_spec = list(set(provided_param_names) - expected_param_names)
        if len(params_not_in_spec) > 0:
            warnings.warn(f"Warning: {params_not_in_spec} in {config_name} will be ignored.", UserWarning)
