# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import uuid
from typing import Mapping, Any

from azureml.data.data_reference import DataReference
from ._pipeline_parameters import PipelineParameter
from ._pipeline_data import PipelineData
from ._module import _OutputBuilder, _InputBuilder
from ._utils import _unique
from ._restclients.designer.models import ModuleDto, GraphDraftEntity


def _input_to_dict(input):
    if isinstance(input, PipelineParameter):
        if input.default_value is not None:
            input = input.default_value
        else:
            # pipeline parameter which has not been assigned
            return {'name': input.name, 'mode': 'parameter'}
    if isinstance(input, DataReference):
        attrs = ["data_reference_name", "path_on_datastore"]
        dict = {attr: input.__getattribute__(attr) for attr in attrs}
        dict["datastore"] = input.datastore.name
    elif hasattr(input, '_registration'):  # registered dataset
        reg = input._registration
        attrs = ["name", "description", "version", "tags", "registered_id", "saved_id"]
        dict = {attr: reg.__getattribute__(attr) for attr in attrs}
    elif hasattr(input, 'dataset'):  # saved dataset
        attrs = ["name", "mode"]
        dict = {attr: input.__getattribute__(attr) for attr in attrs}
        dict["saved_id"] = input.dataset.id
    else:  # direct dataset
        attrs = ["name", "mode"]
        dict = {attr: input.__getattribute__(attr) for attr in attrs}

    return dict


def _module_dto_to_dict(dto: ModuleDto) -> Mapping[str, Any]:
    entity = {}
    if dto.module_version_id is None:
        raise ValueError('No module version id found in module dto.')
    if dto.module_entity is None:
        raise ValueError('No module entity found in module dto: {}.'.format(dto.module_version_id))
    entity["module_id"] = dto.module_version_id
    entity["version"] = dto.module_version
    entity["name"] = dto.module_entity.name
    entity["namespace"] = dto.namespace
    entity["structured_interface"] = {}
    entity["structured_interface"]["inputs"] = [{"name": i.name, "label": i.label, "description": i.description,
                                                "data_type_ids_list": i.data_type_ids_list}
                                                for i in dto.module_entity.structured_interface.inputs]
    entity["structured_interface"]["outputs"] = [{"name": i.name, "label": i.label, "description": i.description,
                                                 "data_type_id": i.data_type_id}
                                                 for i in dto.module_entity.structured_interface.outputs]
    return entity


def _extract_pipeline_definitions(pipelines):
    definitions = []
    for p in pipelines:
        if p._pipeline_definition not in definitions:
            definitions.append(p._pipeline_definition)
    return [d.as_dict() for d in definitions]


class VisualizationContext(object):

    def __init__(self, pipeline_name: str, graph: GraphDraftEntity, module_nodes, pipelines):
        self.pipeline_name = pipeline_name,
        self.graph = graph
        self.module_nodes = module_nodes
        self.pipelines = pipelines


class VisualizationBuilder(object):
    """The builder that constructs visualization info from `azureml.pipeline.wrapper.Pipeline`.
    """
    def __init__(self, context: VisualizationContext):
        self.root_pipeline_name = context.pipeline_name
        self.pipelines = context.pipelines
        self.module_nodes = context.module_nodes
        self.graph = context.graph
        # convert datasources to dict
        all_inputs = [n.inputs[input_name]._get_internal_data_source() for n in self.module_nodes
                      for input_name in n.inputs
                      if n.inputs[input_name].dset is not None]
        inputs = [i for i in all_inputs if not isinstance(i, PipelineData) and not isinstance(i, _OutputBuilder)]
        data_source_to_input_dict = {}
        for input in inputs:
            data_source_to_input_dict[input] = _input_to_dict(input)

        def input_id_func(i):
            if "saved_id" in i:
                return i["saved_id"]
            elif "data_reference_name" in i:
                return i["data_reference_name"]
            else:
                return i["name"]

        # assign nodeId to data sources
        for ds in data_source_to_input_dict.keys():
            input = data_source_to_input_dict[ds]
            input['nodeId'] = str(uuid.uuid3(uuid.NAMESPACE_DNS, input_id_func(input)))

        data_sources = data_source_to_input_dict.values()
        self.data_source_to_input_dict = data_source_to_input_dict
        self.data_sources = _unique(data_sources, input_id_func)

    def build_visualization_dict(self):
        graph = self.graph
        module_node_to_graph_node_mapping = graph.module_node_to_graph_node_mapping

        def serialize_data_node(node, graph_entity):
            def get_dataset_id(data_set_definition):
                if data_set_definition.data_set_reference is not None:
                    return {'dataset_id': data_set_definition.data_set_reference.id}
                elif data_set_definition.saved_data_set_reference is not None:
                    return {'saved_id': data_set_definition.saved_data_set_reference.id}
                else:
                    raise ValueError('Invalid data_set_definition with no dataset in it')

            if node.dataset_id is not None:
                return {'dataset_id': node.dataset_id}
            elif node.data_set_definition.value is None:
                exist = next((x for x in graph_entity.entity_interface.data_path_parameter_list
                              if x.name == node.data_set_definition.parameter_name), None)
                if exist is not None:
                    return get_dataset_id(exist.default_value)
                elif node.data_set_definition.parameter_name is not None:
                    # pipeline parameter which has not been assigned
                    return {'name': node.data_set_definition.parameter_name, 'mode': 'parameter'}
                else:
                    raise ValueError('data_set_definition has no value and no parameter value')
            elif node.data_set_definition.value.literal_value is not None:
                return {'datastore': node.data_set_definition.value.literal_value.data_store_name,
                        'path_on_datastore': node.data_set_definition.value.literal_value.relative_path}
            else:
                return get_dataset_id(node.data_set_definition.value)

        data_references = {}
        from ._graph import SDK_DATA_REFERENCE_NAME
        # TODO: unify this logic
        for node in graph.dataset_nodes:
            if node.data_path_parameter_name is not None:
                data_references[node.data_path_parameter_name] = serialize_data_node(node, graph)
            elif hasattr(node, SDK_DATA_REFERENCE_NAME) and getattr(node, SDK_DATA_REFERENCE_NAME) is not None:
                data_references[getattr(node, SDK_DATA_REFERENCE_NAME)] = serialize_data_node(node, graph)
            elif node.data_set_definition is not None and node.data_set_definition.value is not None \
                    and node.data_set_definition.value.saved_data_set_reference is not None:
                data_references[node.data_set_definition.value.saved_data_set_reference.id] = \
                    serialize_data_node(node, graph)
            else:
                data_references[node.id] = serialize_data_node(node, graph)

        if len(data_references) == 0:
            data_references = None

        steps = {}
        modules = []
        errors = []
        for node in self.module_nodes:
            module_dto = node._module_dto
            modules.append(_module_dto_to_dict(module_dto))

            node_error = node.validate(False)
            errors.append(node_error)

            step_entity = {}
            step_entity = node._serialize_inputs_outputs()
            step_entity["module"] = {"id": module_dto.module_version_id, "version": module_dto.module_version}
            step_entity["validate"] = {
                "error": node_error,
                "module_id": module_dto.module_version_id,
                "namespace": module_dto.namespace,
                "module_name": module_dto.module_name,
                "module_version": module_dto.module_version
            }

            node_id = module_node_to_graph_node_mapping[node._get_instance_id()]
            steps[node_id] = step_entity

        def module_id_func(m):
            return m["module_id"] if(m["version"] is None) else (m["module_id"] + m["version"])

        modules = _unique(modules, module_id_func)

        result = {"pipeline": {"name": self.root_pipeline_name,
                               "data_references": data_references,
                               "steps": steps},
                  "modules": list(modules),
                  "datasources": list(self.data_sources)}
        result.update(self.build_sub_graph_visualization_info())
        return result

    def build_sub_graph_info(self):
        graph = self.graph
        sub_graphs_mapping = []
        module_node_to_graph_node_mapping = graph.module_node_to_graph_node_mapping

        # construct input,output to node mapping dict
        input_to_node = {}
        output_to_node = {}
        for node in self.module_nodes:
            for port_name, port_value in node.inputs.items():
                input_to_node[port_value] = {
                    'portName': node._pythonic_name_to_input_map[port_name],
                    'nodeId': module_node_to_graph_node_mapping[node._get_instance_id()]
                    # TODO: web service
                }
                # if input is from datasource, we should also construct the datasource node mapping here
                if port_value._is_dset_data_source():
                    internal_data = port_value._get_internal_data_source()
                    output_to_node[internal_data] = {
                        'portName': 'output',
                        'nodeId': self.data_source_to_input_dict[internal_data]['nodeId']
                    }
            for port_name, port_value in node.outputs.items():
                output_to_node[port_value] = {
                    'portName': node._pythonic_name_to_output_map[port_name],
                    'nodeId': module_node_to_graph_node_mapping[node._get_instance_id()]
                    # TODO: web service
                }

        # sub graph's input ports can be treated as input's internal for its parent graph
        # or be input's external for its child graph
        # so we add the sub graph's inputs to input_to_node and output_to_node mapping if
        # the inputs is not already in the input_to_node or output_to_node directory
        for p in self.pipelines:
            if p._parent:
                for input_name, input_val in p.inputs.items():
                    if input_val not in input_to_node.keys():
                        input_to_node[input_val] = {
                            'portName': input_name,
                            'nodeId': p._id
                        }
                        output_to_node[input_val] = {
                            'portName': input_name,
                            'nodeId': p._id
                        }

        # pass one: build sub-graph dummy nodes to real nodes mapping and sub graph input ports
        for p in self.pipelines:
            parentGraphId = p._parent._id if p._parent else None

            def extract_internal_value(port_value):
                if isinstance(port_value, _InputBuilder) and isinstance(port_value.dset, _InputBuilder):
                    return port_value.dset
                else:
                    return port_value

            inputs = []
            for input_name, input_val in p.inputs.items():
                input_entity = {}
                input_entity['name'] = input_name
                if parentGraphId is None:
                    input_entity['external'] = []
                    input_entity['internal'] = [input_to_node[input_val]] if input_val in input_to_node.keys() else []
                else:
                    input_entity['external'] = [output_to_node[input_val.dset]] \
                        if input_val.dset in output_to_node.keys() else []
                    # for the nodes inside subgraph, its input maybe wrapped with _InputBuilder
                    input_entity['internal'] = [node for value, node in input_to_node.items()
                                                if
                                                input_val == extract_internal_value(value) and p._id != node['nodeId']]
                inputs.append(input_entity)

            outputs = []
            for output_name, output_val in p.outputs.items():
                output_entity = {}
                output_entity['name'] = output_name
                output_entity['external'] = [node for value, node in input_to_node.items()
                                             if parentGraphId is not None and
                                             output_val == extract_internal_value(value).dset]
                output_entity['internal'] = [output_to_node[output_val]] if output_val in output_to_node.keys() else []
                outputs.append(output_entity)

            sub_graphs_mapping.append({"id": p._id,
                                       "parentGraphId": parentGraphId,
                                       "inputs": inputs,
                                       "outputs": outputs})

        # pass two: mapping real ports to sub-graph dummy ports if possible
        # the re-mapping logic was encapsulated in class SubGraphInfo
        # TODO: currently we will still need this second pass processing only because
        # the output internal/external are not mapped to the right sub graph port
        # which is because the _OutputBuilder is not a self-reference structure like _InputBuilder
        # need to figure out how to avoid this second pass processing in future
        from ._sub_graph_info_builder import SubGraphInfoBuilder
        mapped_sub_graph_infos = []
        for entry in sub_graphs_mapping:
            sub_pipeline = next((p for p in self.pipelines if p._id == entry['id']), None)
            graph = SubGraphInfoBuilder(pipeline=sub_pipeline,
                                        inputs=entry['inputs'], outputs=entry['outputs'],
                                        entry_mapping=sub_graphs_mapping,
                                        module_node_to_graph_node_mapping=module_node_to_graph_node_mapping)
            mapped_sub_graph_infos.append(graph._serialize_to_dict())

        return mapped_sub_graph_infos

    def build_sub_graph_visualization_info(self):
        graph = self.graph

        module_node_to_graph_node_mapping = graph.module_node_to_graph_node_mapping
        node_id_2_graph_id = {}
        for node in self.module_nodes:
            step_id = module_node_to_graph_node_mapping[node._get_instance_id()]
            node_id_2_graph_id[step_id] = node.pipeline._id

        pipeline_entries = self.build_sub_graph_info()
        pipeline_definitions = _extract_pipeline_definitions(self.pipelines)

        return {
            "subGraphInfo": list(pipeline_entries),
            "nodeIdToSubGraphIdMapping": node_id_2_graph_id,
            "subPipelineDefinition": pipeline_definitions
        }

    def build_sub_pipelines_info(self):
        pipeline_dict = self.build_sub_graph_visualization_info()
        sub_graph_info_list = pipeline_dict['subGraphInfo']
        node_id_to_sub_graph_id_mapping = pipeline_dict['nodeIdToSubGraphIdMapping']
        sub_pipeline_definition_list = pipeline_dict['subPipelineDefinition']

        from ._restclients.designer.models import SubGraphInfo, SubPipelineDefinition, SubPipelinesInfo

        sub_graph_info = [SubGraphInfo.from_dict(sub_graph) for sub_graph in sub_graph_info_list]
        sub_pipeline_definition = [SubPipelineDefinition.from_dict(sub_pipeline)
                                   for sub_pipeline in sub_pipeline_definition_list]

        return SubPipelinesInfo(sub_graph_info=sub_graph_info,
                                node_id_to_sub_graph_id_mapping=node_id_to_sub_graph_id_mapping,
                                sub_pipeline_definition=sub_pipeline_definition)
