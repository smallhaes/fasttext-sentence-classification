# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml.core import Workspace
from azureml.data.abstract_dataset import AbstractDataset
from ._restclients.designer.models import ModuleDto


def _get_telemetry_value_from_workspace(workspace: Workspace):
    """
    Get telemetry value out of a Workspace.

    The telemetry values include the following entries:

    * workspace_id
    * workspace_name
    * subscription_id

    :param workspace: The workspace.
    :type workspace: azureml.core.Workspace

    :return: telemetry values.
    :rtype: dict
    """
    telemetry_values = {}
    if workspace is not None:
        telemetry_values['workspace_id'] = workspace._workspace_id
        telemetry_values['workspace_name'] = workspace._workspace_name
        telemetry_values['subscription_id'] = workspace._subscription_id
        try:
            from azureml._base_sdk_common.common import fetch_tenantid_from_aad_token
            telemetry_values['tenant_id'] = fetch_tenantid_from_aad_token(workspace._auth_object._get_arm_token())
        except Exception as e:
            telemetry_values['tenant_id'] = "Error retrieving tenant id: {}".format(e)

    return telemetry_values


def _get_telemetry_value_from_module_dto(module_dto: ModuleDto):
    """
    Get telemetry value out of a module dto.

    The telemetry values include the following entries:

    * module_id
    * module_type
    * step_type

    :param module_dto: The module dto.

    :return: telemetry values.
    :rtype: dict
    """
    telemetry_values = {}
    if module_dto is not None:
        telemetry_values['module_id'] = module_dto.module_entity.id
        telemetry_values['step_type'] = module_dto.module_entity.step_type

        # ModuleScope.global
        if module_dto.module_scope == '1':
            telemetry_values['module_type'] = 'Global'
        # ModuleScope.workspace
        elif module_dto.module_scope == '2':
            telemetry_values['module_type'] = 'Custom'
        # ModuleScope.anonymous
        elif module_dto.module_scope == '3':
            telemetry_values['module_type'] = 'Anonymous'
            telemetry_values['module_id'] = module_dto.module_version_id
        else:
            telemetry_values['module_type'] = 'Unknown'

    return telemetry_values


def _get_telemetry_value_from_pipeline(pipeline, pipeline_parameters=None, compute_target=None,
                                       data_sources=None, sub_pipelines_info=None, on_create=False):
    """
    Get telemetry value out of a pipeline.

    The telemetry values include the following entries:

    * pipeline_id: A uuid generated for each pipeline created.
    * defined_by: The way the pipeline is created, using @dsl.pipeline or raw code.
    * node_count: The total count of all module nodes.
    * pipeline_parameters_count: The total count of all pipeline parameters.
    * data_pipeline_parameters_count: The total count of all pipeline parameters that are dataset.
    * literal_pipeline_parameters_count: The total count of all pipeline parameters that are literal values.
    * input_count: The total count of data sources.
    * compute_count: The total count of distinct computes.
    * compute_type_count: The total count of distinct compute types.
    * top_level_node_count: The total count of top level nodes & pipelines.
    * subpipeline_count: The total count of sub pipelines.

    :param pipeline: The pipeline.
    :param pipeline_parameters: The pipeline parameters.
    :param compute_target: The compute target.
    :param data_sources: Data sources of the pipeline.
    :param sub_pipelines_info: Sub pipeline infos of the pipeline.
    :param on_create: Whether the pipeline was just created, which means compute target, pipeline parameters, etc
                      are not available.
    :return: telemetry values.
    :rtype: dict
    """
    telemetry_values = _get_telemetry_value_from_workspace(pipeline.workspace)
    all_nodes, _ = pipeline._expand_pipeline_nodes()
    telemetry_values['pipeline_id'] = pipeline._id
    telemetry_values['defined_by'] = "dsl" if pipeline._use_dsl else "raw"
    telemetry_values['node_count'] = len(all_nodes)
    telemetry_values['top_level_node_count'] = len(pipeline.nodes)
    if on_create:
        # We do not have enough information to populate all telemetry values.
        return telemetry_values

    telemetry_values.update(_get_telemetry_value_from_pipeline_parameter(pipeline_parameters))

    if compute_target is not None:
        compute_set = set([node._resolve_compute(compute_target)[0] for node in all_nodes])
        compute_type_set = set([node._resolve_compute(compute_target)[1] for node in all_nodes])
        telemetry_values['compute_count'] = len(compute_set)
        telemetry_values['compute_type_count'] = len(compute_type_set)

    if data_sources is not None:
        telemetry_values['input_count'] = len(data_sources)
    if sub_pipelines_info is not None:
        telemetry_values['subpipeline_count'] = len(sub_pipelines_info.sub_graph_info) - 1

    return telemetry_values


def _get_telemetry_value_from_pipeline_parameter(pipeline_parameters):
    telemetry_values = {}
    pipeline_parameters_count = 0
    data_pipeline_parameters_count = 0
    literal_pipeline_parameters_count = 0

    if pipeline_parameters is not None:
        pipeline_parameters_count = len(pipeline_parameters)
        data_pipeline_parameters_count = len([x for x in pipeline_parameters.values() if
                                              isinstance(x, AbstractDataset)])
        literal_pipeline_parameters_count = pipeline_parameters_count - data_pipeline_parameters_count

    telemetry_values['pipeline_parameters_count'] = pipeline_parameters_count
    telemetry_values['data_pipeline_parameters_count'] = data_pipeline_parameters_count
    telemetry_values['literal_pipeline_parameters_count'] = literal_pipeline_parameters_count
    return telemetry_values


def _get_telemetry_value_from_module(module, compute_target=None, additional_value=None):
    """
    Get telemetry value out of a Module.

    The telemetry values include the following entries:

    * load_source: The source type which the module node is loaded.
    * specify_input_mode: Whether the input mode is being by users.
    * specify_output_mode: Whether the output mode is being by users.
    * specify_output_datastore: Whether the output datastore is specified by users.
    * pipeline_id: the pipeline_id if the module node belongs to some pipeline.
    * specify_node_level_compute: Whether the node level compute is specified by users.
    * compute_type: The compute type that the module uses.

    :param pipeline: The pipeline.
    :param pipeline_parameters: The pipeline parameters.
    :param compute_target: The compute target.
    :return: telemetry values.
    :rtype: dict
    """
    telemetry_values = {}
    telemetry_values.update(_get_telemetry_value_from_workspace(module._workspace))
    telemetry_values.update(_get_telemetry_value_from_module_dto(module._module_dto))
    telemetry_values['load_source'] = module._load_source
    telemetry_values['specify_input_mode'] = module._specify_input_mode
    telemetry_values['specify_output_mode'] = module._specify_output_mode
    telemetry_values['specify_output_datastore'] = module._specify_output_datastore

    node_compute_target, specify_node_level_compute = module._resolve_compute(compute_target) \
        if compute_target is not None else (None, False)

    if hasattr(module, 'pipeline'):
        telemetry_values['pipeline_id'] = module.pipeline._id
    if node_compute_target is not None:
        telemetry_values['specify_node_level_compute'] = specify_node_level_compute
        telemetry_values['compute_type'] = node_compute_target[1]

    telemetry_values.update(additional_value or {})
    return telemetry_values
