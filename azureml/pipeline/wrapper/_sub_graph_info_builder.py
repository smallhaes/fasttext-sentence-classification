# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import copy
from typing import List, Mapping
from azureml.data._dataset import AbstractDataset
from ._pipeline import Pipeline
from ._module import Module, _InputBuilder
from ._pipeline_parameters import PipelineParameter
from ._restclients.designer.models import SubGraphInfo, ComputeSetting, DatastoreSetting, DataPathParameter,\
    SubGraphPortInfo, SubGraphParameterAssignment, SubGraphDataPathParameterAssignment, \
    SubPipelineParameterAssignment, Parameter, SubPipelineDefinition, Kwarg


def _is_default_compute_node(node):
    if isinstance(node, Module):
        return node.runsettings.target is None
    return False


def _is_default_datastore_node(node, default_datastore):
    if isinstance(node, Module):
        non_default_datastore_output = next((output for output in node.outputs.values()
                                            if output.datastore != default_datastore), None)
        return non_default_datastore_output is None
    return False


def _get_node_parameters_para_mapping(node):
    if isinstance(node, Module):
        return node._build_params().items()
    else:
        return node._parameters_param.items()


def _get_data_store_setting(default_data_store):
    from azureml.data.abstract_datastore import AbstractDatastore
    if isinstance(default_data_store, str):
        return DatastoreSetting(data_store_name=default_data_store)
    elif isinstance(default_data_store, AbstractDatastore):
        return DatastoreSetting(data_store_name=default_data_store.name)
    else:
        return None


def _get_compute_setting(default_compute_target):
    if default_compute_target is None:
        return None
    elif isinstance(default_compute_target, str):
        return ComputeSetting(name=default_compute_target)
    elif isinstance(default_compute_target, tuple):
        if len(default_compute_target) == 2:
            return ComputeSetting(name=default_compute_target[0])
            # TODO: how to set proper compute_type
            # compute_type=default_compute_target[1])
    else:
        return ComputeSetting(name=default_compute_target.name)
        # compute_type=default_compute_target.type)


def _correct_default_compute_target(sub_pipeline_definition, default_compute_target):
    sub_pipeline_definition.default_compute_target = _get_compute_setting(default_compute_target)


def _correct_default_data_store(sub_pipeline_definition, default_data_store):
    sub_pipeline_definition.default_data_store = _get_data_store_setting(default_data_store)


def _normalize_from_module_name(from_module_name):
    """return the bottom module file name
    if from_module_name = 'some_module', return 'some_module'
    if from_module_name = 'some_module.sub_module', return 'sub_module'
    """
    if from_module_name is None:
        return None

    try:
        import re
        entries = re.split(r'[.]', from_module_name)
        return entries[-1]
    except Exception:
        return None


def _build_sub_pipeline_definition(name, description,
                                   default_compute_target, default_data_store,
                                   id, parent_definition_id=None,
                                   from_module_name=None, parameters=None, func_name=None):
    def parameter_to_kv(parameter):
        from inspect import Parameter
        key = parameter.name
        value = parameter.default if parameter.default is not Parameter.empty else None
        kv = Kwarg(key=key, value=value)
        return kv

    compute_target = _get_compute_setting(default_compute_target)
    data_store = _get_data_store_setting(default_data_store)
    parameter_list = [] if parameters is None else [parameter_to_kv(p) for p in parameters]

    return SubPipelineDefinition(name=name, description=description,
                                 default_compute_target=compute_target, default_data_store=data_store,
                                 pipeline_function_name=func_name,
                                 id=id, parent_definition_id=parent_definition_id,
                                 from_module_name=_normalize_from_module_name(from_module_name),
                                 parameter_list=parameter_list)


class SubGraphInfoBuilder:
    """
    This class is a wrapper for sub graph infos.
    And it encapsulates the mapping logic from real node ports to sub graph dummy node ports

    :param pipeline: The pipeline of the current sub graph
    :type pipeline: Pipeline
    :param inputs: All input ports of the current sub graph
    :type inputs: List[Mapping]
    :param outputs: All output ports of the current sub graph
    :type outputs: List[Mapping]
    :param entry_mapping: The real node ports to sub graph dummy node ports mapping
    :type entry_mapping: List[Mapping]
    :param module_node_to_graph_node_mapping: module node id to graph node id mapping
    :type module_node_to_graph_node_mapping: Mapping
    """
    def __init__(self, pipeline: Pipeline,
                 inputs: List[Mapping] = None,
                 outputs: List[Mapping] = None,
                 entry_mapping: List[Mapping] = None,
                 module_node_to_graph_node_mapping: Mapping = None):

        self._pipeline = pipeline
        self._name = pipeline.name
        self._description = pipeline.description
        self._defaultCompute = pipeline._get_default_compute_target()
        self._defaultDatastore = pipeline.default_datastore
        self._id = pipeline._id
        self._parentGraphId = pipeline._parent._id if pipeline._parent else None
        self._pipelineDefinitionId = pipeline._pipeline_definition.id
        self._module_node_to_graph_node_mapping = module_node_to_graph_node_mapping

        # deepcopy inputs and outputs to avoid original mapping pollution
        if inputs is not None:
            self._inputs = copy.deepcopy(inputs)
        else:
            self._inputs = []
        if outputs is not None:
            self._outputs = copy.deepcopy(outputs)
        else:
            self._outputs = []

        self._entry_mapping = entry_mapping

        if entry_mapping is not None:
            self._map_to_dummy_port()

    def _serialize_to_dict(self):
        compute = _get_compute_setting(self._defaultCompute)
        datastore = _get_data_store_setting(self._defaultDatastore)

        sub_graph_default_compute_target_nodes = [self._get_graph_node_id(node)
                                                  for node in self._pipeline.nodes
                                                  if _is_default_compute_node(node)]

        sub_graph_default_data_store_nodes = [self._get_graph_node_id(node)
                                              for node in self._pipeline.nodes
                                              if _is_default_datastore_node(node, self._pipeline._default_datastore)]

        sub_graph_parameter_assignment = self._get_parameter_assignments()
        sub_graph_data_path_parameter_assignment = self._get_data_parameter_assignments()

        sub_graph_info = SubGraphInfo(
            name=self._name, description=self._description,
            default_compute_target=compute, default_data_store=datastore,
            id=self._id, parent_graph_id=self._parentGraphId,
            pipeline_definition_id=self._pipelineDefinitionId,
            sub_graph_parameter_assignment=sub_graph_parameter_assignment,
            sub_graph_data_path_parameter_assignment=sub_graph_data_path_parameter_assignment,
            sub_graph_default_compute_target_nodes=sub_graph_default_compute_target_nodes,
            sub_graph_default_data_store_nodes=sub_graph_default_data_store_nodes,
            inputs=[SubGraphPortInfo.from_dict(i) for i in self._inputs],
            outputs=[SubGraphPortInfo.from_dict(o) for o in self._outputs])

        return sub_graph_info.as_dict()

    def _get_ports(self, direction, direction_index, scope):
        if direction == 'inputs':
            return self._inputs[direction_index][scope]
        elif direction == 'outputs':
            return self._outputs[direction_index][scope]
        else:
            return []

    def _set_port(self, direction, direction_index, scope, scope_index, value):
        if direction == 'inputs':
            self._inputs[direction_index][scope][scope_index] = value
        elif direction == 'outputs':
            self._outputs[direction_index][scope][scope_index] = value

    def _map_to_sub_graph_dummy_port_op(self, direction, index, scope, sub_graph_candidates,
                                        sub_graph_direction, sub_graph_scope):
        for i in range(len(self._get_ports(direction, index, scope))):
            cur_node = self._get_ports(direction, index, scope)[i]
            for sub_graph in sub_graph_candidates:
                dummy = next((port for port in sub_graph[sub_graph_direction]
                             if any(sub_port for sub_port in port[sub_graph_scope]
                                    if(sub_port['nodeId'] == cur_node['nodeId'] and
                                       sub_port['portName'] == cur_node['portName']))), None)
                if dummy is not None:
                    self._set_port(direction,
                                   index,
                                   scope,
                                   i,
                                   {'nodeId': sub_graph['id'], 'portName': dummy['name']})

    def _map_inputs_internal_to_dummy_port(self, i):
        if (len(self._inputs[i]['internal']) == 0):
            return

        graph_candidates = [graph for graph in self._entry_mapping if graph['parentGraphId'] == self._id]
        self._map_to_sub_graph_dummy_port_op('inputs', i, 'internal', graph_candidates, 'inputs', 'internal')

    def _map_outputs_internal_to_dummy_port(self, i):
        if (len(self._outputs[i]['internal']) == 0):
            return

        graph_candidates = [graph for graph in self._entry_mapping if graph['parentGraphId'] == self._id]
        self._map_to_sub_graph_dummy_port_op('outputs', i, 'internal', graph_candidates, 'outputs', 'internal')

    def _map_inputs_external_to_dummy_port(self, i):
        if (len(self._inputs[i]['external']) == 0):
            return

        graph_candidates = [graph for graph in self._entry_mapping if graph['parentGraphId'] == self._parentGraphId and
                            graph['id'] != self._id]
        self._map_to_sub_graph_dummy_port_op('inputs', i, 'external', graph_candidates, 'outputs', 'internal')

        parent_graph = [graph for graph in self._entry_mapping if graph['id'] == self._parentGraphId]
        self._map_to_sub_graph_dummy_port_op('inputs', i, 'external', parent_graph, 'inputs', 'external')

    def _map_outputs_external_to_dummy_port(self, i):
        if (len(self._outputs[i]['external']) == 0):
            return

        graph_candidates = [graph for graph in self._entry_mapping if graph['parentGraphId'] == self._parentGraphId and
                            graph['id'] != self._id]
        self._map_to_sub_graph_dummy_port_op('outputs', i, 'external', graph_candidates, 'inputs', 'internal')

        parent_graph = [graph for graph in self._entry_mapping if graph['id'] == self._parentGraphId]
        self._map_to_sub_graph_dummy_port_op('outputs', i, 'external', parent_graph, 'outputs', 'external')

    def _map_to_dummy_port(self):
        for i in range(len(self._inputs)):
            self._map_inputs_internal_to_dummy_port(i)
            self._map_inputs_external_to_dummy_port(i)

        for i in range(len(self._outputs)):
            self._map_outputs_internal_to_dummy_port(i)
            self._map_outputs_external_to_dummy_port(i)

    def _get_graph_node_id(self, node):
        if isinstance(node, Pipeline):
            return node._id
        else:
            return self._module_node_to_graph_node_mapping[node._get_instance_id()]

    def _get_parameter_assignments(self):
        parameter_assignments = []
        parameter_assignments_mapping = {}
        for k, v in self._pipeline._parameters_param.items():
            parameter_assignments_mapping[k] = []

        def find_matched_parent_input(input_value):
            is_sub_pipeline = self._pipeline._is_sub_pipeline
            for input_k, input_v in self._pipeline.inputs.items():
                if (not is_sub_pipeline and input_value.dset == input_v.dset) or \
                   (is_sub_pipeline and input_value.dset == input_v):
                    return True
            return False

        def get_parent_parameter_name(node, para_v):
            # for sub pipeline, it should be a PipelineParameter wrapped with _InputBuilder
            if isinstance(node, Pipeline) and isinstance(para_v, _InputBuilder) and \
                    isinstance(para_v._get_internal_data_source(), PipelineParameter):
                return para_v.dset.name
            elif isinstance(node, Module) and isinstance(para_v, _InputBuilder) and \
                    isinstance(para_v._get_internal_data_source(), PipelineParameter):
                return para_v.name
            elif isinstance(node, Module) and isinstance(para_v, PipelineParameter):
                return para_v.name
            else:
                return None

        def try_to_add_assignments(assignments_mapping, parent_param_name, node_id, para_name):
            if parent_param_name in parameter_assignments_mapping.keys():
                assignments = parameter_assignments_mapping[parent_param_name]
                assignments.append(
                    SubPipelineParameterAssignment(node_id=node_id,
                                                   parameter_name=para_name))

        for node in self._pipeline.nodes:
            for input_name, input_value in node.inputs.items():
                if find_matched_parent_input(input_value):
                    try_to_add_assignments(assignments_mapping=parameter_assignments_mapping,
                                           parent_param_name=input_value.dset.name,
                                           node_id=self._get_graph_node_id(node),
                                           para_name=input_name)

            for para_k, para_v in _get_node_parameters_para_mapping(node):
                try_to_add_assignments(assignments_mapping=parameter_assignments_mapping,
                                       parent_param_name=get_parent_parameter_name(node, para_v),
                                       node_id=self._get_graph_node_id(node),
                                       para_name=para_k)

        for k, v in self._pipeline._parameters_param.items():
            if len(parameter_assignments_mapping[k]) > 0:
                parameter = Parameter(name=k, default_value=v.default_value) \
                    if isinstance(v, PipelineParameter) else Parameter(name=k)
                parameter_assignments.append(SubGraphParameterAssignment(
                    parameter=parameter,
                    parameter_assignments=parameter_assignments_mapping[k]))

        return parameter_assignments

    def _get_data_parameter_assignments(self):
        dataset_parameter_assignment = []

        for input_k, input_v in self._pipeline.inputs.items():
            # dataset parameter assignment
            if isinstance(input_v, _InputBuilder) and isinstance(input_v.dset, AbstractDataset):
                from ._graph import _GraphEntityBuilder
                dataset_def = _GraphEntityBuilder._get_dataset_def_from_dataset(input_v.dset)
                data_set_parameter = DataPathParameter(
                    name=input_k,
                    default_value=dataset_def.value,
                    is_optional=False,
                    data_type_id='DataFrameDirectory')
                dataset_parameter_assignment.append(SubGraphDataPathParameterAssignment(
                    data_set_path_parameter=data_set_parameter,
                    # currently, we don't need to know which node is assigned
                    data_set_path_parameter_assignments=[]))

        return dataset_parameter_assignment
