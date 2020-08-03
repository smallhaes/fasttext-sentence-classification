# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml.core.runconfig import RunConfiguration
from azureml._base_sdk_common.field_info import _FieldInfo
from ._restclients.service_caller_factory import _DesignerServiceCallerFactory
from ._restclients.designer.models import RunSettingParameter, ComputeRunSettingParameter
from ._dynamic import KwParameter, create_kw_function_from_parameters
from ._module_validator import ModuleValidator, ValidationError


class _RunSettings(object):
    def __init__(self, params: [RunSettingParameter], module_name, workspace):
        self._target = None
        for p in params:
            if p.is_compute_target:
                self.target = p.default_value
            else:
                setattr(self, p.argument_name, p.default_value)
        self._params_spec = {p.argument_name: p for p in params}
        self._workspace = workspace
        self._generate_configure_func(module_name)

    def _generate_configure_func(self, module_name):
        func_docstring_lines = []
        func_docstring_lines.append(f"Run setting configuration for module [{module_name}]")
        if len(self._params_spec) > 0:
            func_docstring_lines.append("\n")
        params, _doc_string = _format_params(list(self._params_spec.values()))
        func_docstring_lines.extend(_doc_string)
        func_docstring = '\n'.join(func_docstring_lines)
        self.__doc__ = func_docstring

        def create_run_setting_configure_func(**kwargs):
            for k, v in kwargs.items():
                ModuleValidator.validate_single_runsettings_parameter(k, v, self,
                                                                      _RunSettings._process_error)
                setattr(self, k, v)

        self.configure = create_kw_function_from_parameters(
            create_run_setting_configure_func,
            documentation=func_docstring,
            parameters=params,
            func_name='configure'
        )

    def configure(self):
        """
        Configure the runsettings.

        Note that this method will be replaced by a dynamic generated one at runtime with parameters
        that corresponds to the runsettings of the module.
        """
        pass

    def __repr__(self):
        params_str = ''
        param_names = self._params_spec.keys()
        for name in param_names:
            params_str += '{}: {}\n'.format(name, getattr(self, name))
        return params_str

    @property
    def use_default_compute(self):
        return self._target is None

    @property
    def target(self):
        """
        The name of compute parameter in HDInsight module is "Compute Name", while other modules are "Target".
        So we use "target" here to provide a common access for compute parameter.
        """

        return self._target

    @target.setter
    def target(self, compute):
        if compute is not None:
            if isinstance(compute, str):
                # Get compute type
                service_caller = _DesignerServiceCallerFactory.get_instance(self._workspace)
                compute_list = service_caller.list_experiment_computes(include_test_types=True)
                target_compute = next((c for c in compute_list if c.name == compute), None)
                if target_compute is None:
                    # Force fetch computes
                    compute_list = service_caller.list_experiment_computes(include_test_types=True,
                                                                           use_sdk_cache=False)
                    target_compute = next((c for c in compute_list if c.name == compute), None)
                    if target_compute is None:
                        raise ValueError(f"Cannot find compute '{compute}' in workspace")
                else:
                    compute = (compute, target_compute.compute_type)
            elif not (isinstance(compute, tuple) and len(compute) == 2 and
                      isinstance(compute[0], str) and isinstance(compute[1], str)):
                raise ValueError("Bad value for target, expect compute_name: "
                                 "string or (compute_name: string, compute_type: string)")
        self._target = compute

    @staticmethod
    def _process_error(e: Exception, error_type):
        # Raise exception when hit INVALID_RUNSETTING_PARAMETER
        # Since missing some parameters is allowed when calling runsettings.configure(), for example:
        # runsettings.configure(node_count=2)
        # runsettings.target = 'amlcompute'
        if error_type == ValidationError.INVALID_RUNSETTING_PARAMETER:
            raise e


class _K8sRunSettings(object):
    def __init__(self, params: [ComputeRunSettingParameter]):
        sections = {}
        for p in params:
            if p.section_argument_name not in sections:
                sections[p.section_argument_name] = []
            sections[p.section_argument_name].append(p)
        self._params_spec = sections
        for section_name in sections:
            setattr(self, section_name, _K8sRunSettingsSection(sections[section_name], section_name))

    def _doc_string(self):
        return f"Configuration sections: {[s for s in self._params_spec]}."

    def __repr__(self):
        params_str = ''
        params = self._params_spec
        for section in params:
            params_str += '{}:\n'.format(section)
            for p in params[section]:
                params_str += '\t{}: {}\n'.format(p.argument_name, getattr(getattr(self, section), p.argument_name))
        return params_str


class _K8sRunSettingsSection(object):
    def __init__(self, params: [ComputeRunSettingParameter], section_name):
        self._section_name = section_name
        for p in params:
            setattr(self, p.argument_name, p.default_value)
        self._params_spec = {p.argument_name: p for p in params}
        self._generate_configure_func(params)

    def _generate_configure_func(self, params):
        func_docstring_lines = [params[0].section_description]
        func_params, _doc_string = _format_params(params)
        func_docstring_lines.extend(_doc_string)
        func_docstring = '\n'.join(func_docstring_lines)
        self.__doc__ = func_docstring

        def create_compute_runsetting_configure(**kwargs):
            for k, v in kwargs.items():
                ModuleValidator.validate_single_k8srunsettings_parameter(k, v, self,
                                                                         _RunSettings._process_error)
                setattr(self, k, v)

        self.configure = create_kw_function_from_parameters(create_compute_runsetting_configure,
                                                            documentation=func_docstring,
                                                            parameters=func_params,
                                                            func_name='configure')

    def configure(self):
        """
        Configure the runsettings.

        Note that this method will be replaced by a dynamic generated one at runtime with parameters
        that corresponds to the runsettings of the module.
        """
        pass

    def __repr__(self):
        params_str = ''
        param_names = self._params_spec.keys()
        for name in param_names:
            params_str += '{}: {}\n'.format(name, getattr(self, name))
        return params_str


def set_compute_field(k8srunsettings, compute_type, run_config):
    compute_params_spec = k8srunsettings._params_spec
    if compute_params_spec is None:
        return
    compute_field_map = {'Cmk8s': 'cmk8scompute', 'CmAks': 'cmakscompute'}
    if compute_type in compute_field_map:
        field_name = compute_field_map[compute_type]
        aks_config = {'configuration': dict()}
        for section_name in compute_params_spec:
            for param in compute_params_spec[section_name]:
                value = getattr(getattr(k8srunsettings, section_name), param.argument_name)
                if value is not None:
                    aks_config['configuration'][param.argument_name] = value
        run_config._initialized = False
        setattr(run_config, field_name, aks_config)
        run_config._initialized = True
        RunConfiguration._field_to_info_dict[field_name] = _FieldInfo(dict,
                                                                      "{} specific details.".format(field_name))
        run_config.history.output_collection = True


def _format_params(source_params):
    target_params = []
    func_docstring_lines = []
    # The default value in spec is not match the value in description, so we remove "default value" part in doc string
    # for ComputeRunSettingParameter
    is_compute_run_settings = isinstance(source_params[0], ComputeRunSettingParameter)
    for param in source_params:
        param_name = param.argument_name
        param_name_in_doc = param_name
        # For Hdi module, the name of target parameter is "compute_name"
        # But in sdk, we use "target" to access the target parameter
        # So we add a hint here to indicate "target" is exactly "compute_name"
        if getattr(param, 'is_compute_target', None) is True and param_name != 'target':
            param_name = 'target'
            param_name_in_doc = f'target ({param_name_in_doc})'
        if param.is_optional:
            func_docstring_lines.append(":param {}: {} (optional{})"
                                        .format(param_name_in_doc, param.description,
                                                "" if is_compute_run_settings else ", default value is {}."
                                                .format(param.default_value)))
        else:
            func_docstring_lines.append(":param {}: {}".format(param_name_in_doc, param.description))
        parameter_type = param.parameter_type
        func_docstring_lines.append(":type {}: {}".format(param_name_in_doc, parameter_type.value))
        target_params.append(KwParameter(
            param_name,
            annotation=param.description,
            default=param.default_value,
            _type=parameter_type.value))
    return target_params, func_docstring_lines
