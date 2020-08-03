# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import uuid
from typing import List
from ._module import Module, _InputBuilder, _OutputBuilder, _AttrDict
from ._module_dto import _python_type_to_type_code
from azureml.data._dataset import _Dataset, AbstractDataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from ._pipeline_data import PipelineData
from ._pipeline_parameters import PipelineParameter
from ._dataset import _GlobalDataset
from ._restclients.designer.models import GraphDraftEntity, GraphModuleNode, GraphDatasetNode, \
    GraphEdge, ParameterAssignment, PortInfo, DataSetDefinition, RegisteredDataSetReference, SavedDataSetReference, \
    DataSetDefinitionValue, EntityInterface, Parameter, DataPath, DataPathParameter, \
    OutputSetting, GraphModuleNodeRunSetting, RunSettingParameterAssignment, ComputeSetting, \
    RunSettingUIWidgetTypeEnum, ComputeType, ComputeSettingMlcComputeInfo
from azureml.core.compute import AmlCompute, ComputeInstance, RemoteCompute, HDInsightCompute

SDK_DATA_REFERENCE_NAME = 'sdk_data_reference_name'


def _topology_sort_nodes(nodes: List[Module]):
    """
    Sort the modules in the topological order
    If there has circlular dependencies among the modules, the returned list order is not assured

    :param: nodes: list of modules
    :type nodes: List[Module]
    :return list of modules in topological order
    : rtype: List[Module]
    """
    from ._pipeline import Pipeline

    node_len = len(nodes)
    if node_len == 0:
        return []

    output_to_module = {}
    visted = {}
    stack = []
    for node in nodes:
        visted[node] = False
        for output in node.outputs.values():
            output_to_module[output] = node

    def get_dependencies(node, visted):
        dependencies = []
        for input in node.inputs.values():
            if isinstance(input, _InputBuilder):
                dset = input._get_internal_data_source()
                if isinstance(dset, _OutputBuilder):
                    if dset in output_to_module.keys():
                        dependencies.append(output_to_module[dset])
                elif isinstance(dset, Module) or isinstance(dset, Pipeline):
                    for output in dset.outputs.values():
                        if isinstance(output, _OutputBuilder) and output in output_to_module.keys():
                            dependencies.append(output_to_module[output])
                elif isinstance(dset, _AttrDict):
                    for output in dset.values():
                        if isinstance(output, _OutputBuilder) and output in output_to_module.keys():
                            dependencies.append(output_to_module[output])
        return [d for d in dependencies if not visted[d]]

    result = []
    for node in nodes:
        if not visted[node]:
            visted[node] = True
            stack.append(node)
            while len(stack) != 0:
                cur = stack[-1]
                dependencies = get_dependencies(cur, visted)
                if (len(dependencies) == 0):
                    result.append(cur)
                    stack.pop()
                else:
                    for d in dependencies:
                        if not visted[d]:
                            stack.append(d)
                            visted[d] = True

    return result


class _GraphEntityBuilderContext(object):
    def __init__(self, compute_target=None, pipeline_parameters=None, pipeline_regenerate_outputs=None,
                 module_nodes=None, workspace=None, default_datastore=None):
        """
        Init the context needed for graph builder.

        :param compute_target: The compute target.
        :type compute_target: tuple(name, type)
        :param pipeline_parameters: The pipeline parameters.
        :type pipeline_parameters: dict
        :param pipeline_regenerate_outputs: the `regenerate_output` value of all module node
        :type pipeline_regenerate_outputs: bool
        """
        self.compute_target = compute_target
        self.pipeline_parameters = pipeline_parameters
        self.pipeline_regenerate_outputs = pipeline_regenerate_outputs

        self.module_nodes = module_nodes
        self.workspace = workspace
        self.default_datastore = default_datastore


class _GraphEntityBuilder(object):
    """The builder that constructs SMT graph-related entities from `azureml.pipeline.wrapper.Module`."""
    DATASOURCE_PORT_NAME = 'data'

    def __init__(self, context: _GraphEntityBuilderContext):
        self._context = context
        self._modules = _topology_sort_nodes(context.module_nodes)
        self._nodes = {}
        self._input_nodes = {}
        self._data_path_parameter_input = {}

    def build_graph_entity(self, is_local_run=False):
        """Build graph entity that can be used to create pipeline draft and pipeline run.

        :param is_local_run: whether module execute in local
        :type is_local_run: bool
        :return Tuple of (graph entity, module node run settings, dataset definition value assignments)
        :rtype tuple
        """

        graph_entity = GraphDraftEntity()
        module_node_to_graph_node_mapping = {}

        # Prepare the entity
        graph_entity.dataset_nodes = []
        graph_entity.module_nodes = []
        graph_entity.edges = []
        graph_entity.entity_interface: EntityInterface = EntityInterface(parameters=[], data_path_parameters=[],
                                                                         data_path_parameter_list=[])

        if self._context.compute_target is not None:
            default_compute_name, default_compute_type = self._context.compute_target
            graph_entity.default_compute = ComputeSetting(name=default_compute_name,
                                                          compute_type=ComputeType.mlc,
                                                          mlc_compute_info=ComputeSettingMlcComputeInfo(
                                                              mlc_compute_type=default_compute_type))

        module_node_run_settings = []

        # Note that the modules must be sorted in topological order
        # So that the dependent outputs are built before we build the inputs_map
        for module in self._modules:
            module_node = self._build_graph_module_node(module,
                                                        self._context.pipeline_regenerate_outputs,
                                                        module_node_to_graph_node_mapping)
            graph_entity.module_nodes.append(module_node)
            self._nodes[module_node.id] = module_node

            # Note that outputs_map must be build for edges to have the correct producer information.
            outputs_map = module._build_outputs_map(
                producer=module_node, default_datastore=self._context.default_datastore)
            inputs_map = module._build_inputs_map()
            for input_name, i in inputs_map.items():
                edge = None
                if isinstance(i, DatasetConsumptionConfig) or isinstance(i, _GlobalDataset) \
                        or isinstance(i, PipelineParameter):
                    dataset_node = self._get_or_create_dataset_node(graph_entity, module, i)
                    edge = self._produce_edge_dataset_node_to_module_node(input_name, dataset_node, module_node)
                elif isinstance(i, PipelineData):
                    edge = self._produce_edge_module_node_to_module_node(input_name, i, module_node)
                else:
                    raise ValueError("Invalid input type: {0}".format(type(i)))
                if edge is not None:
                    graph_entity.edges.append(edge)

            module_node_run_settings.append(
                self._produce_module_runsettings(module, module_node))

            self._update_module_node_params(graph_entity, module_node, module,
                                            inputs_map, outputs_map, self._context.pipeline_parameters)

        self._update_data_path_parameter_list(graph_entity, self._context.pipeline_parameters)
        setattr(graph_entity, 'module_node_to_graph_node_mapping', module_node_to_graph_node_mapping)

        return graph_entity, module_node_run_settings

    def _produce_module_runsettings(self, module: Module, module_node: GraphModuleNode):
        module_dto = module._module_dto
        if module_dto is None:
            raise ValueError("No module_dto found for module.")

        use_default_compute = module.runsettings.use_default_compute

        # do not remove this, or else module_node_run_setting does not make a difference
        module_node.use_graph_default_compute = use_default_compute
        module_node_run_setting = GraphModuleNodeRunSetting()
        module_node_run_setting.module_id = module_dto.module_version_id
        module_node_run_setting.node_id = module_node.id
        module_node_run_setting.step_type = module_dto.module_entity.step_type
        module_node_run_setting.run_settings = []

        runsettings = module._runsettings
        k8srunsettings = module._k8srunsettings
        params_spec = runsettings._params_spec
        for param_name in params_spec:
            param = params_spec[param_name]
            if param.is_compute_target:
                compute_run_settings = []
                # Always add compute settings
                # Since module may use default compute, we don't have to detect this, MT will handle
                if k8srunsettings is not None:
                    for section_name in k8srunsettings._params_spec:
                        for compute_param in k8srunsettings._params_spec[section_name]:
                            compute_param_value = getattr(getattr(k8srunsettings, section_name),
                                                          compute_param.argument_name)
                            compute_run_settings.append(RunSettingParameterAssignment(name=compute_param.name,
                                                                                      value=compute_param_value,
                                                                                      value_type=0))
                compute_name = runsettings.target
                compute_type = None
                if isinstance(compute_name, tuple):
                    compute_name, compute_type = runsettings.target
                module_node_run_setting.run_settings.append(
                    RunSettingParameterAssignment(name=param.name, value=compute_name, value_type=0,
                                                  use_graph_default_compute=runsettings.use_default_compute,
                                                  mlc_compute_type=compute_type,
                                                  compute_run_settings=compute_run_settings))
            else:
                param_value = getattr(runsettings, param_name)
                module_node_run_setting.run_settings.append(
                    RunSettingParameterAssignment(name=param.name, value=param_value, value_type=0))
        return module_node_run_setting

    def _produce_edge_dataset_node_to_module_node(self, input_name, dataset_node, module_node):
        source = PortInfo(node_id=dataset_node.id, port_name=self.DATASOURCE_PORT_NAME)
        dest = PortInfo(node_id=module_node.id, port_name=input_name)
        return GraphEdge(source_output_port=source, destination_input_port=dest)

    def _produce_edge_module_node_to_module_node(self, input_name, pipeline_data: PipelineData, dest_module_node):
        source_module_node = pipeline_data._producer
        source = PortInfo(node_id=source_module_node.id, port_name=pipeline_data._port_name)
        dest = PortInfo(node_id=dest_module_node.id, port_name=input_name)
        return GraphEdge(source_output_port=source, destination_input_port=dest)

    def _get_or_create_dataset_node(self, graph_entity: GraphDraftEntity, module: Module, input):
        if input in self._input_nodes:
            return self._input_nodes[input]
        else:
            dataset_node = self._build_graph_datasource_node(input, module)
            self._input_nodes[input] = dataset_node
            self._nodes[dataset_node.id] = dataset_node
            graph_entity.dataset_nodes.append(dataset_node)
            return dataset_node

    def _build_graph_module_node(self, module: Module,
                                 pipeline_regenerate_outputs: bool,
                                 module_node_to_graph_node_mapping) -> GraphModuleNode:
        node_id = self._generate_node_id()
        module_dto = module._module_dto
        regenerate_output = pipeline_regenerate_outputs \
            if pipeline_regenerate_outputs is not None else module.regenerate_output
        module_node = GraphModuleNode(id=node_id, module_id=module_dto.module_version_id,
                                      regenerate_output=regenerate_output)
        module_node.module_parameters = []
        module_node.module_metadata_parameters = []
        module_node_to_graph_node_mapping[module._get_instance_id()] = node_id
        return module_node

    def _update_module_node_params(self, graph_entity: GraphDraftEntity, module_node: GraphModuleNode, module: Module,
                                   inputs_map, outputs_map, pipeline_parameters):
        module_dto = module._module_dto
        interface = module_dto.module_entity.structured_interface

        parameters = interface.parameters
        node_parameters = {}
        node_pipeline_parameters = {}

        for p in parameters:
            node_parameters[p.name] = p.default_value

        user_provided_params = module._build_params()

        for param_name, param_value in user_provided_params.items():
            # TODO: Use an enum for value_type
            is_sub_pipeline_parameter = module.pipeline is not None and module.pipeline._is_sub_pipeline
            # only promote the parameter as graph's PipelineParameter when it does not belong to a sub-pipeline
            if not is_sub_pipeline_parameter and isinstance(param_value, PipelineParameter):
                # value is of type PipelineParameter, use its name property
                # TODO parameter assignment expects 'Literal', 'GraphParameterName', 'Concatenate', 'Input'??
                node_pipeline_parameters[param_name] = param_value.name
                type_code = _python_type_to_type_code(type(param_value.default_value))
                exist = next((x for x in graph_entity.entity_interface.parameters
                              if x.name == param_value.name), None) is not None
                if not exist:
                    name = param_value.name
                    value = param_value.default_value
                    # Check if user choose to override with pipeline parameters
                    if pipeline_parameters is not None and len(pipeline_parameters) > 0:
                        for k, v in pipeline_parameters.items():
                            if k == name:
                                value = v
                    graph_entity.entity_interface.parameters.append(Parameter(
                        name=name,
                        default_value=value,
                        is_optional=False,
                        type=type_code))

                # Pipeline parameters should not be included in module parameters
                if param_name in node_parameters:
                    del node_parameters[param_name]
            elif isinstance(param_value, PipelineParameter):
                node_parameters[param_name] = param_value.default_value
            elif isinstance(param_value, _InputBuilder):
                internal_data = param_value._get_internal_data_source()
                value = internal_data.default_value if isinstance(internal_data, PipelineParameter) else internal_data
                node_parameters[param_name] = value
            else:
                node_parameters[param_name] = param_value

        for _, input in inputs_map.items():
            if input in self._data_path_parameter_input.values():
                continue
            if isinstance(input, PipelineParameter):
                self._data_path_parameter_input[input.name] = input

        self._batch_append_module_node_parameter(module_node, node_parameters)
        self._batch_append_module_node_pipeline_parameters(module_node, node_pipeline_parameters)

        module_node.module_output_settings = []
        for output in outputs_map.values():
            output_setting = OutputSetting(name=output.name, data_store_name=output.datastore.name,
                                           data_store_mode=output._output_mode,
                                           path_on_compute=output._output_path_on_compute,
                                           overwrite=output._output_overwrite,
                                           data_reference_name=output.name)
            module_node.module_output_settings.append(output_setting)

    def _update_data_path_parameter_list(self, graph_entity: GraphDraftEntity, pipeline_parameters):

        def get_override_parameters_def(name, origin_val, pipeline_parameters):
            # Check if user choose to override with pipeline parameters
            if pipeline_parameters is not None and len(pipeline_parameters) > 0:
                for k, v in pipeline_parameters.items():
                    if k == name:
                        if isinstance(v, _GlobalDataset) or isinstance(v, AbstractDataset):
                            return self._get_dataset_def_from_dataset(v)
                        else:
                            raise ValueError('Invalid parameter value for dataset parameter: {0}'.format(k))

            return origin_val

        for name, pipeline_parameter in self._data_path_parameter_input.items():
            dset = pipeline_parameter.default_value
            dataset_def = None

            if isinstance(dset, AbstractDataset):
                dset = dset.as_named_input(name).as_mount()

            if isinstance(dset, DatasetConsumptionConfig):
                dataset_consumption_config = dset
                dataset = dataset_consumption_config.dataset
                dataset._ensure_saved(self._context.workspace)
                dataset_def = self._get_dataset_def_from_dataset(dataset)

            dataset_def = get_override_parameters_def(name, dataset_def, pipeline_parameters)
            if dataset_def is not None:
                exist = next((x for x in graph_entity.entity_interface.data_path_parameter_list
                              if x.name == name), None) is not None
                if not exist:
                    graph_entity.entity_interface.data_path_parameter_list.append(DataPathParameter(
                        name=name,
                        default_value=dataset_def.value,
                        is_optional=False,
                        data_type_id='DataFrameDirectory'
                    ))

    def _batch_append_module_node_pipeline_parameters(self, module_node: GraphModuleNode, params):
        for k, v in params.items():
            param_assignment = ParameterAssignment(name=k, value=v, value_type=1)
            module_node.module_parameters.append(param_assignment)

    def _batch_append_module_node_parameter(self, module_node: GraphModuleNode, params):
        for k, v in params.items():
            param_assignment = ParameterAssignment(name=k, value=v, value_type=0)
            module_node.module_parameters.append(param_assignment)

    def _append_module_meta_parameter(self, module_node: GraphModuleNode, param_name, param_value):
        param_assignment = ParameterAssignment(name=param_name, value=param_value, value_type=0)
        module_node.module_metadata_parameters.append(param_assignment)

    def _build_graph_datasource_node(self, input, module: Module) -> GraphDatasetNode:
        node_id = self._generate_node_id()
        # set attribute SDK_DATA_REFERENCE_NAME only for sdk generated graph json
        if isinstance(input, DatasetConsumptionConfig) and isinstance(input.dataset, _Dataset):
            input.dataset._ensure_saved(self._context.workspace)
            dataset_def = self._get_dataset_def_from_dataset(input.dataset)
            data_node = GraphDatasetNode(id=node_id, data_set_definition=dataset_def)
            setattr(data_node, SDK_DATA_REFERENCE_NAME, input.dataset.name)
            return data_node

        if isinstance(input, PipelineParameter):
            dataset_def = DataSetDefinition(data_type_short_name="DataFrameDirectory",
                                            parameter_name=input.name)
            return GraphDatasetNode(id=node_id, data_set_definition=dataset_def)

        if isinstance(input, _GlobalDataset):
            dataset_def = self._get_dataset_def_from_dataset(input)
            data_node = GraphDatasetNode(id=node_id, data_set_definition=dataset_def)
            setattr(data_node, SDK_DATA_REFERENCE_NAME, input.data_reference_name)
            return data_node

    @staticmethod
    def _extract_mlc_compute_type(target_type):
        if target_type == AmlCompute._compute_type or target_type == RemoteCompute._compute_type or \
                target_type == HDInsightCompute._compute_type or target_type == ComputeInstance._compute_type:
            if target_type == AmlCompute._compute_type:
                return 'AmlCompute'
            elif target_type == ComputeInstance._compute_type:
                return 'ComputeInstance'
            elif target_type == HDInsightCompute._compute_type:
                return 'Hdi'
        return None

    @staticmethod
    def _get_dataset_def_from_dataset(dataset):
        if isinstance(dataset, _GlobalDataset):
            data_path = DataPath(data_store_name=dataset.data_store_name, relative_path=dataset.relative_path)
            dataset_def_val = DataSetDefinitionValue(literal_value=data_path)
            dataset_def = DataSetDefinition(
                data_type_short_name='DataFrameDirectory',
                value=dataset_def_val
            )
            return dataset_def

        if dataset._registration and dataset._registration.registered_id:
            dataset_ref = RegisteredDataSetReference(
                id=dataset._registration and dataset._registration.registered_id,
                version=dataset._registration and dataset._registration.version
            )
            dataset_def_val = DataSetDefinitionValue(data_set_reference=dataset_ref)
        else:
            saved_id = dataset.id
            if saved_id:
                saved_dataset_ref = SavedDataSetReference(id=saved_id)
                dataset_def_val = DataSetDefinitionValue(saved_data_set_reference=saved_dataset_ref)

        dataset_def = DataSetDefinition(
            data_type_short_name='AnyDirectory',
            value=dataset_def_val
        )
        return dataset_def

    def _generate_node_id(self) -> str:
        """
        Generate an 8-character node Id.

        :return: node_id
        :rtype: str
        """
        guid = str(uuid.uuid4())
        id_len = 8
        while guid[:id_len] in self._nodes:
            guid = str(uuid.uuid4())

        return guid[:id_len]


def _int_str_to_run_setting_ui_widget_type_enum(int_str_value):
    return list(RunSettingUIWidgetTypeEnum)[int(int_str_value)]
