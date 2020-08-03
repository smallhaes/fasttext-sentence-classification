# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Defines the class for creating reusable Azure Machine Learning workflows."""

import datetime as dt
import uuid
import os
import tempfile
from pathlib import Path
from inspect import signature
from typing import List, Union, Mapping, Callable

from azureml.core import Datastore
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.data._dataset import _Dataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data.file_dataset import FileDataset
from azureml._base_sdk_common._docstring_wrapper import experimental
from azureml.exceptions._azureml_exception import UserErrorException

from ._module import Module, _AttrDict, _OutputBuilder, _InputBuilder
from ._pipeline_parameters import PipelineParameter
from ._loggerfactory import _LoggerFactory, _PUBLIC_API, track
from ._dataset import _GlobalDataset
from ._pipeline_run_orchestrator import _orchestrate_pipeline_run, STEP_PREFIX, NODE_ID, WORKING_DIR
from ._telemetry import _get_telemetry_value_from_pipeline, _get_telemetry_value_from_module
from ._utils import _is_prod_workspace, _in_jupyter_nb
from ._backend_provider import SmtBackendProvider, PipelineProviderContext, PublishedPipelineProviderContext
from ._graph import _GraphEntityBuilder, _GraphEntityBuilderContext
from ._visualize import VisualizationBuilder, VisualizationContext
from ._module_validator import ModuleValidator
from ._pipeline_validator import PipelineValidator, ValidationError

_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


def _unify_input_port_name(node_name, node_id, port_name, port_value):
    """get input port's unified name

    if the port is corresponded to a subgraph's pipeline parameter, take it as the parameter name
    otherwise, take it as {node_name}:{port_name}

    :param node_name: name of the node where the port is
    :type node_name: str
    :param node_id: id of the node where the port is
    :type node_id: str
    :param port_name: port's name
    :type port_name: str
    :param port_value: the port's input
    :type: obj
    """
    if isinstance(port_value, _InputBuilder):
        # if it is _InputBuilder type, that means it comes from a subgraph's pipeline parameter
        if isinstance(port_value._dset, _InputBuilder):
            return f'{port_value._dset.name}'
        elif isinstance(port_value._dset, _GlobalDataset):
            return f'{port_value._dset.data_reference_name}_{node_id}'
        elif isinstance(port_value._dset, _Dataset):
            return f'{port_value._dset.name}_{node_id}'
        elif isinstance(port_value._dset, PipelineParameter):
            return f'{port_value._dset.name}'
        else:
            return f'{node_name}:{port_name}'
    else:
        return f'{node_name}:{port_name}'


def _extract_input_port_value(port_value):
    """extract the underlying _InputBuilder
    This is needed when the input comes from sub graph's pipeline parameter

    :param port_value: the port's input
    :type port_value: obj
    """
    if isinstance(port_value, _InputBuilder):
        if isinstance(port_value._dset, _InputBuilder):
            return port_value._dset
        else:
            return port_value
    else:
        return port_value


def _expand_pipeline_to_pipelines(pipeline, pipelines, parent=None):
    """Expand the pipeline into list.
    """
    pipelines.append(pipeline)
    pipeline._parent = parent
    for node in pipeline.nodes:
        if isinstance(node, Pipeline):
            _expand_pipeline_to_pipelines(node, pipelines, pipeline)


class Pipeline:
    """
    Represents a collection of steps which can be executed as a reusable Azure Machine Learning workflow.

    Use a Pipeline to create and manage workflows that stitch together various machine learning
    phases. Each machine learning phase, such as data preparation and model training, can consist of one or
    more `azureml.pipeline.wrapper.Module` nodes in a Pipeline.

    :param nodes: The nodes of module used to create the pipeline.
    :type nodes: list[azureml.pipeline.wrapper.Module
            or azureml.pipeline.wrapper.Pipeline]
    :param outputs: The pipeline outputs.
    :type outputs: dict
    :param workspace: The workspace of the piepline
    :type workspace: azureml.core.Workspace
    :param name: The name of the pipeline
    :type name: str
    :param description: The description of the pipeline
    :type description: str
    :param default_compute_target: The compute target of built pipeline.
        May be a compute target object or the string name of a compute target on the workspace.
        The priority of compute target assignment goes: module's run config > sub pipeline's default compute target >
        parent pipeline's default compute target.
        Optionally, if the compute target is not available at pipeline creation time, you may specify a tuple of
       ('compute target name', 'compute target type') to avoid fetching the compute target object(AmlCompute
        type is 'AmlCompute' and RemoteCompute type is 'VirtualMachine').
    :type default_compute_target: azureml.core.compute.DsvmCompute
                        or azureml.core.compute.AmlCompute
                        or azureml.core.compute.RemoteCompute
                        or azureml.core.compute.HDInsightCompute
                        or str
                        or tuple
    :param default_datastore: The default datastore of pipeline.
    :type default_datastore: str or azureml.core.Datastore
    """

    def __init__(self, nodes: List[Union[Module, 'Pipeline']], outputs: Mapping[str, _OutputBuilder] = None,
                 workspace=None, name=None, description=None,
                 default_compute_target=None, default_datastore=None, _use_dsl=False):
        """
        Initialize Pipeline.

        :param nodes: The nodes of module used to create the pipeline.
        :type nodes: list[azureml.pipeline.wrapper.Module
            or azureml.pipeline.wrapper.Pipeline]
        :param outputs: The pipeline outputs.
        :type outputs: dict
        :param workspace: The workspace of the piepline
        :type workspace: azureml.core.Workspace
        :param name: The name of the pipeline
        :type name: str
        :param description: The description of the pipeline
        :type description: str
        :param default_compute_target: The compute target of built pipeline.
            May be a compute target object or the string name of a compute target on the workspace.
            The priority of compute target assignment goes: module's run config >
            sub pipeline's default compute target > parent pipeline's default compute target.
            Optionally, if the compute target is not available at pipeline creation time, you may specify a tuple of
        ('compute target name', 'compute target type') to avoid fetching the compute target object(AmlCompute
            type is 'AmlCompute' and RemoteCompute type is 'VirtualMachine').
        :type default_compute_target: azureml.core.compute.DsvmCompute
                            or azureml.core.compute.AmlCompute
                            or azureml.core.compute.RemoteCompute
                            or azureml.core.compute.HDInsightCompute
                            or str
                            or tuple
        :param default_datastore: The default datastore of pipeline.
        :type default_datastore: str or azureml.core.Datastore
        :param _use_dsl: Whether created by @dsl.pipeline
        :type _use_dsl: bool
        """
        self._id = str(uuid.uuid4())
        self.nodes = tuple(nodes)
        if len(nodes) != len(set(nodes)):
            raise ValueError('Could not add duplicate nodes to pipeline.')

        self._set_inputs()

        if outputs is None:
            self._set_outputs({})
        else:
            self._set_outputs(outputs)

        self._workspace = workspace
        self._default_datastore = default_datastore
        self._name = name
        self._description = description
        self._parent = None

        self._default_compute_target = default_compute_target

        self._backend_provider = SmtBackendProvider()

        self._parameters_param = {}

        # add current pipeline into current dsl pipeline if there is one
        from .dsl.pipeline import _try_to_add_node_to_current_pipeline, _is_pipeline_stack_empty
        _try_to_add_node_to_current_pipeline(self)
        self._is_sub_pipeline = False if _is_pipeline_stack_empty() else True

        # add pipeline definition, if it is wrapped with dsl, the definition will be overwritten in pipeline_decorator
        from ._sub_graph_info_builder import _build_sub_pipeline_definition
        self._pipeline_definition = _build_sub_pipeline_definition(
            name=name,
            description=description,
            default_compute_target=self._get_default_compute_target(),
            default_data_store=self.default_datastore,
            id=str(uuid.uuid4()))

        self._use_dsl = _use_dsl
        # If we are using dsl. Skip this.
        if not _use_dsl:
            _LoggerFactory.trace(_get_logger(), "Pipeline_created", self._get_telemetry_values(on_create=True))

    @property
    def name(self):
        """
        Get or set the name of the Pipeline.

        :return: The name.
        :rtype: str
        """
        if self._name is None:
            now = dt.datetime.now()
            self._name = f'Pipeline-Created-on-{now.month}-{now.day}-{now.year}'
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def description(self):
        """
        Get or set the description of the Pipeline.

        :return: The description.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @property
    def inputs(self):
        """
        Get the inputs of the Pipeline.

        :return: pipeline inputs.
        :rtype: dict
        """
        return self._inputs

    @property
    def outputs(self):
        """
        Get the outputs of the Module.

        :return: pipeline outputs.
        :rtype: dict
        """
        return self._outputs

    @property
    def workspace(self):
        """
        Get the workspace of the Pipeline.

        This will check if all nodes in pipeline are from the same workspace.

        :return: The workspace.
        :rtype: azureml.core.Workspace
        """
        for node in self.nodes:
            new_workspace = node.workspace
            if new_workspace is None:
                continue

            if self._workspace is None:
                self._workspace = new_workspace
            else:
                assert self._workspace._workspace_id == new_workspace._workspace_id, \
                    f'Not all pipeline nodes are from the same workspace: {self._workspace}, {new_workspace}'

        return self._workspace

    @property
    def default_datastore(self):
        """
        Get the default datastore of the pipeline.

        :return: the default datastore.
        :rtype: azureml.core.Datastore
        """
        if self._default_datastore is None or isinstance(self._default_datastore, str):
            ws = self.workspace
            if isinstance(self._default_datastore, str) and ws is not None:
                self._default_datastore = Datastore(ws, name=self._default_datastore)
            elif ws is not None:
                self._default_datastore = ws.get_default_datastore()
        return self._default_datastore

    def _set_inputs(self):
        """
        Setter method to set pipeline inputs.

        All inputs from external module will be current pipeline's input.
        """
        all_pipeline_node_outputs = [output for node in self.nodes for output_name, output in node.outputs.items()]
        # append all nodes, since node with one output could be used as input as well
        all_pipeline_node_outputs.extend([node for node in self.nodes])
        # append all nodes' outputs, since node's outputs _AttrDict with one output could be used as input as well
        all_pipeline_node_outputs.extend([node.outputs for node in self.nodes])

        inputs = {}
        for node in self.nodes:
            for input_name, input in node.inputs.items():
                if input._dset and input._dset not in all_pipeline_node_outputs and \
                        not isinstance(input._dset, _GlobalDataset) and \
                        not isinstance(input._dset, _Dataset) and \
                        not isinstance(input._dset, DatasetConsumptionConfig) and \
                        not isinstance(input._dset, FileDataset):
                    instance_id = node._get_instance_id() if isinstance(node, Module) else node._id
                    inputs[_unify_input_port_name(node.name, instance_id, input_name, input)] = \
                        _extract_input_port_value(input)
        self._inputs = _AttrDict(**inputs)

    def _set_outputs(self, outputs: Mapping[str, _OutputBuilder]):
        """
        Setter method to set pipeline outputs, will check if right type of outputs is passed.
        """
        error_msg = "The return type of decorated function should be a mapping from dataset name to " \
                    "azureml.pipeline.wrapper.Module._OutputBuilder"
        assert isinstance(outputs, dict), error_msg
        for key, value in outputs.items():
            assert isinstance(key, str), error_msg
            assert isinstance(value, _OutputBuilder), error_msg
        self._outputs = _AttrDict(**outputs)

    def _build_pipeline_func_parameters(self, func, args, kwargs):
        """
        build the pipeline func parameter mapping
        """
        def all_p(parameters):
            for value in parameters.values():
                yield value

        parameters = all_p(signature(func).parameters)
        for arg in args:
            self._parameters_param[parameters.__next__().name] = arg
        for k, v in kwargs.items():
            self._parameters_param[k] = v

    def _add_node(self, node: Union[Module, 'Pipeline']):
        """
        Add a node into pipeline, type of node could be Module or Pipeline.

        :param node:
        :type azureml.pipeline.wrapper.Module or azureml.pipeline.wrapper.Pipeline
        :return:
        """
        if node in self.nodes:
            raise ValueError('node already exists.')
        self.nodes += (node,)

    def _get_default_compute_target(self, default_compute_target=None):
        """
        Try to resovle the default compute target to tupe(compute_name, compute_type).

        :param default_compute_target
        :type str or AmlCompute or tuple(str, str)
        :return:
        """
        if default_compute_target is None:
            default_compute_target = self._default_compute_target

        if default_compute_target is None:
            return None, "AmlCompute"

        # try to resolve compute target
        if isinstance(default_compute_target, str):
            if self.workspace is None:
                # this should only happens in dsl pipeline, when we initialize a Pipeline with no nodes
                return default_compute_target, "AmlCompute"
            _targets = self.workspace.compute_targets
            target = _targets.get(default_compute_target)
            if target is None:
                print(default_compute_target + " not found in workspace, assume this is an AmlCompute")
                return default_compute_target, "AmlCompute"
            else:
                return target.name, target.type
        elif isinstance(default_compute_target, tuple):
            if not len(default_compute_target) == 2:
                raise ValueError('Compute target tuple must have 2 elements (compute name, compute type)')
            return default_compute_target
        else:
            return default_compute_target.name, default_compute_target.type

    @track(_get_logger, activity_type=_PUBLIC_API)
    def submit(self, experiment_name, default_compute_target=None, description=None, pipeline_parameters=None,
               tags=None, continue_on_step_failure=None, regenerate_outputs=None):
        """
        Submit current pipeline run to workspace.

        :param experiment_name: The experiment name
        :type experiment_name: str
        :param default_compute_target: The default compute target used to run pipeline
        :type default_compute_target: str
        :param description: The description of the submitted pipeline run
        :type description: str
        :param pipeline_parameters: An optional dictionary of pipeline parameter assignments for the PipelineDraft
        :type pipeline_parameters: dict({str:str})
        :param tags: Tags to be added to the submitted run, {"tag": "value"}
        :type tags: dict
        :param continue_on_step_failure: Indicates whether to continue pipeline execution if a step fails.
            If True, only steps that have no dependency on the output of the failed step will continue execution.
        :type continue_on_step_failure: bool
        :param regenerate_outputs: Indicates whether to force regeneration of all step outputs and disallow data
            reuse for this run. If False, this run may reuse results from previous runs and subsequent runs may reuse
            the results of this run.
        :type regenerate_outputs: bool

        :return: run
        :rtype: azureml.pipeline.wrapper.pipeline_run.PipelineRun
        """
        # validate pipeline
        graphyaml = self._build_visualization_dict()
        self._validate(graphyaml, fail_fast=True)

        workspace = self.workspace
        default_compute_target = self._get_default_compute_target(default_compute_target)

        module_nodes, _ = self._expand_pipeline_nodes()
        pipelines = self._expand_pipeline_to_pipelines()
        graph_builder_context = _GraphEntityBuilderContext(compute_target=default_compute_target,
                                                           pipeline_parameters=pipeline_parameters,
                                                           pipeline_regenerate_outputs=regenerate_outputs,
                                                           module_nodes=module_nodes,
                                                           workspace=workspace,
                                                           default_datastore=self.default_datastore)

        graph_entity_builder = _GraphEntityBuilder(graph_builder_context)
        graph, module_node_run_settings = graph_entity_builder.build_graph_entity()

        visualization_context = VisualizationContext(pipeline_name=self.name,
                                                     graph=graph,
                                                     module_nodes=module_nodes,
                                                     pipelines=pipelines)
        visualization_builder = VisualizationBuilder(visualization_context)
        sub_pipelines_info = visualization_builder.build_sub_pipelines_info()

        context = PipelineProviderContext(
            experiment_name=experiment_name,
            graph=graph,
            sub_pipelines_info=sub_pipelines_info,
            module_node_run_settings=module_node_run_settings,
            compute_target=default_compute_target,
            pipeline_parameters=pipeline_parameters,
            continue_on_step_failure=continue_on_step_failure,
            regenerate_outputs=regenerate_outputs,
            submit_description=description,
            tags=tags,
        )

        run = self._backend_provider.submit_pipeline(workspace=workspace,
                                                     pipeline=self, context=context)

        telemetry_value = self._get_telemetry_values(
            pipeline_parameters=pipeline_parameters,
            compute_target=default_compute_target,
            data_sources=visualization_builder.data_sources,
            sub_pipelines_info=sub_pipelines_info,
            additional_value={
                'run_id': run.id,
            })

        _LoggerFactory.trace(_get_logger(), "Pipeline_submit", telemetry_value)
        for node in module_nodes:
            _LoggerFactory.trace(_get_logger(),
                                 "Pipeline_submit_module",
                                 _get_telemetry_value_from_module(node, default_compute_target, {
                                     'run_id': run.id,
                                 }),
                                 adhere_custom_dimensions=False)

        run.visualization_builder = visualization_builder

        return run

    @track(_get_logger, activity_type=_PUBLIC_API)
    def save(self, experiment_name, id=None, default_compute_target=None,
             pipeline_parameters=None, tags=None, properties=None):
        """
        Save pipeline as PipelineDraft.

        :param experiment_name: The experiment name for the PipelineDraft;
        :type experiment_name: str
        :param id: Existing pipeline draft id. If specified, pipeline will be save to that pipeline draft.
        :type id: str
        :param default_compute_target: he default compute target used to run pipeline
        :type default_compute_target: str
        :param pipeline_parameters: An optional dictionary of pipeline parameter assignments for the PipelineDraft.
        :type pipeline_parameters: dict({str:str})
        :param tags: Tags to be added to the submitted run, {"tag": "value"}
        :type tags: dict
        :param properties: Optional properties dictionary for the PipelineDraft,
            only needed when saving as a new PipelineDraft
        :type properties: dict({str:str})
        :return: The created PipelineDraft.
        :rtype: azureml.pipeline.core.PipelineDraft
        """
        workspace = self.workspace
        default_compute_target = self._get_default_compute_target(default_compute_target)

        module_nodes, _ = self._expand_pipeline_nodes()
        pipelines = self._expand_pipeline_to_pipelines()
        graph_builder_context = _GraphEntityBuilderContext(compute_target=default_compute_target,
                                                           pipeline_parameters=pipeline_parameters,
                                                           module_nodes=module_nodes,
                                                           workspace=workspace,
                                                           default_datastore=self.default_datastore)

        graph_entity_builder = _GraphEntityBuilder(graph_builder_context)
        graph, module_node_run_settings = graph_entity_builder.build_graph_entity()

        visualization_context = VisualizationContext(pipeline_name=self.name,
                                                     graph=graph,
                                                     module_nodes=module_nodes,
                                                     pipelines=pipelines)
        visualization_builder = VisualizationBuilder(visualization_context)
        sub_pipelines_info = visualization_builder.build_sub_pipelines_info()

        context = PipelineProviderContext(
            experiment_name=experiment_name,
            graph=graph,
            sub_pipelines_info=sub_pipelines_info,
            module_node_run_settings=module_node_run_settings,
            compute_target=default_compute_target,
            pipeline_parameters=pipeline_parameters,
            tags=tags,
            properties=properties
        )

        telemetry_value = self._get_telemetry_values(
            pipeline_parameters=pipeline_parameters,
            compute_target=default_compute_target,
            data_sources=visualization_builder.data_sources,
            sub_pipelines_info=sub_pipelines_info,
            additional_value={
                'draft_id': id if id is not None else ''
            })

        _LoggerFactory.trace(_get_logger(), "Pipeline_save", telemetry_value)

        return self._backend_provider.save_pipeline_as_draft(_id=id, workspace=workspace,
                                                             pipeline=self, context=context)

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Pipeline_publish')
    def publish(self, experiment_name: str, name: str, description: str = None,
                parameters=None, tags=None):
        """
        Publish a pipeline and make it available for rerunning.

        You can get the pipeline rest endpoint from the PublishedPipeline object returned by this function. With the
        rest endpoint, you can invoke the pipeline from external applications using REST calls. For information
        about how to authenticate when calling REST endpoints, see https://aka.ms/pl-restep-auth.

        The original pipeline associated with the pipeline run is used as the base for the published pipeline.

        :param experiment_name: The name of the published pipeline's experiment.
        :type experiment_name: str
        :param name: The name of the published pipeline.
        :type name: str
        :param description: The description of the published pipeline.
        :type description: str
        :param parameters: parameters of published pipeline.
        :type parameters: dict[str, str]
        :param tags: tags of pipeline to publish
        :type tags: dict[str, str]

        :return: Created published pipeline.
        :rtype: azureml.pipeline.wrapper._published_pipeline.PublishedPipeline
        """
        graph_builder_context = _GraphEntityBuilderContext(
            compute_target=self._get_default_compute_target(),
            module_nodes=self._expand_pipeline_nodes()[0],
            workspace=self.workspace,
            default_datastore=self.default_datastore,
            pipeline_parameters=parameters)

        graph_entity_builder = _GraphEntityBuilder(graph_builder_context)
        graph, _ = graph_entity_builder.build_graph_entity()
        context = PublishedPipelineProviderContext(
            pipeline_name=name, graph=graph,
            pipeline_description=description, tags=tags, use_pipeline_endpoint=False,
            pipeline_parameters=parameters, experiment_name=experiment_name, )
        result = self._backend_provider. \
            publish_pipeline(workspace=self.workspace, context=context, pipeline=self)
        telemetry_values = self._get_telemetry_values(pipeline_parameters=parameters)
        telemetry_values.update({
            'pipeline_id': result.id,
            'use_pipeline_endpoint': False,
        })
        _LoggerFactory.trace(_get_logger(), "Pipeline_publish", telemetry_values)
        return result

    @experimental
    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Pipeline_publish')
    def publish_to_endpoint(self, experiment_name, name: str, pipeline_endpoint_name: str,
                            description: str = None, pipeline_endpoint_description: str = None,
                            set_as_default: bool = True, use_existing_pipeline_endpoint: bool = True,
                            tags: dict = None, parameters=None):
        """
        Publish a pipeline to pipeline_endpoint.

        A pipeline enpoint is a :class:`azureml.pipeline.wrapper.Pipeline` workflow
         that can be triggered from a unique endpoint URL.

        :param experiment_name: The name of the published pipeline's experiment.
        :type experiment_name: str
        :param name: The name of the published pipeline.
        :type name: str
        :param description: The description of the published pipeline.
        :type description: str
        :param pipeline_endpoint_name: The name of pipeline endpoint.
        :type pipeline_endpoint_name: str
        :param pipeline_endpoint_description: The description of pipeline endpoint.
        :type pipeline_endpoint_description: str
        :param set_as_default: Whether to use pipeline published as the default version of pipeline endpoint.
        :type set_as_default: bool
        :param use_existing_pipeline_endpoint: Whether to use existing pipeline endpoint.
        :type use_existing_pipeline_endpoint: bool
        :param tags: tags of pipeline to publish
        :type tags: dict[str, str]
        :param parameters: parameters of published pipeline.
        :type parameters: dict[str, str]

        :return: Created published pipeline inside pipeline endpoint.
        :rtype: azureml.pipeline.wrapper._published_pipeline.PublishedPipeline
        """
        graph_builder_context = _GraphEntityBuilderContext(
            compute_target=self._get_default_compute_target(),
            module_nodes=self._expand_pipeline_nodes()[0],
            workspace=self.workspace,
            default_datastore=self.default_datastore,
            pipeline_parameters=parameters)

        graph_entity_builder = _GraphEntityBuilder(graph_builder_context)
        graph, _ = graph_entity_builder.build_graph_entity()
        context = PublishedPipelineProviderContext(
            experiment_name=experiment_name, graph=graph,
            pipeline_name=name, pipeline_description=description,
            pipeline_endpoint_name=pipeline_endpoint_name,
            pipeline_endpoint_description=pipeline_endpoint_description,
            tags=tags, set_as_default=set_as_default,
            use_existing_pipeline_endpoint=use_existing_pipeline_endpoint,
            use_pipeline_endpoint=True, pipeline_parameters=parameters, )
        result = self._backend_provider. \
            publish_pipeline(workspace=self.workspace, context=context, pipeline=self)
        telemetry_values = self._get_telemetry_values(pipeline_parameters=parameters)
        telemetry_values.update({
            'pipeline_id': result.id,
            'use_pipeline_endpoint': True,
            'set_as_default': set_as_default,
            'use_existing_pipeline_endpoint': use_existing_pipeline_endpoint,
        })
        _LoggerFactory.trace(_get_logger(), "Pipeline_publish", telemetry_values)
        return result

    @track(_get_logger, activity_type=_PUBLIC_API)
    def validate(self, fail_fast=False):
        """
        Graph/module validation and visualization.

        :param fail_fast: Whether the validation process will fail-fast
        :rtype fail_fast: bool

        :return: List of errors
        :rtype: list
        """
        graphyaml = self._build_visualization_dict()

        if _in_jupyter_nb():
            from ._widgets._visualize import _visualize
            is_prod = _is_prod_workspace(self.workspace)
            envinfo = {
                "subscription_id": self.workspace.subscription_id
            }
            _visualize(graphyaml, envinfo=envinfo, is_prod=is_prod)

        validate_result = self._validate(graphyaml=graphyaml, fail_fast=fail_fast)

        return validate_result

    def _validate(self, graphyaml, fail_fast):
        pipeline_steps = graphyaml['pipeline']['steps']
        errors = []

        def process_cycle_error(cycle):
            cycles_nodes = ["{0}({1})".format(pipeline_steps[node.node_id]['validate']['module_name'], node.node_id)
                            for node in cycle]
            error = ValidationError(message="Module cycle detected, including nodes: {}".format(cycles_nodes),
                                    error_type=ValidationError.MODULE_CYCLE)
            if fail_fast:
                raise error
            else:
                errors.append({'error': [
                    {'message': error.message,
                     'type': error.error_type}
                ]})

        PipelineValidator.validate_pipeline_steps(pipeline_steps, errors)
        PipelineValidator.validate_module_cycle(pipeline_steps, process_cycle_error)

        result = "validation passed"
        if len(errors) > 0:
            result = "validation failed"

        telemetry_value = self._get_telemetry_values(additional_value={
            'validation_passed': len(errors) == 0
        })

        _LoggerFactory.trace(_get_logger(), "Pipeline_validate", telemetry_value)
        if len(errors) > 0:
            for module_errors in errors:
                if 'ModuleCycle' in module_errors['error'][0]['type']:
                    pass
                else:
                    module_info = {
                        'module_id': module_errors['module_id'],
                        'module_version': module_errors['module_version'],
                    }
                    for one_error in module_errors['error']:
                        if 'type' in one_error:
                            telemetry_value = self._get_telemetry_values()
                            telemetry_value.update(module_info)
                            telemetry_value.update({
                                'error_message': one_error['message'],
                                'error_type': one_error['type']
                            })
                            _LoggerFactory.trace(_get_logger(), "Pipeline_module_validate_error", telemetry_value,
                                                 adhere_custom_dimensions=False)

        return {
            "result": result,
            "errors": errors
        }

    def _get_telemetry_values(self, pipeline_parameters=None, compute_target=None, data_sources=None,
                              sub_pipelines_info=None, on_create=False, additional_value=None):
        telemetry_values = _get_telemetry_value_from_pipeline(self,
                                                              pipeline_parameters=pipeline_parameters,
                                                              compute_target=compute_target,
                                                              data_sources=data_sources,
                                                              sub_pipelines_info=sub_pipelines_info,
                                                              on_create=on_create)
        if additional_value is not None:
            telemetry_values.update(additional_value)

        return telemetry_values

    def _replace_module(self, old_module: Module, new_module: Module,
                        recursive: bool):
        if recursive:
            nodes, _ = self._expand_pipeline_nodes()
        else:
            nodes = self.nodes
        for node in nodes:
            if isinstance(node, Module) and node._is_target_module(old_module):
                # replace target node's module_version
                node._replace(new_module)

    @track(_get_logger, activity_type=_PUBLIC_API)
    def replace(self, old_module_func: Callable, new_module_func: Callable,
                recursive=False, force=False):
        """
        replace modules by module_function

        :param old_module_func: a module function which can generate the old module you want to replace
        :type old_module_func: function
        :param new_module_func: a module function which can generate the new module to replace the old one
        :type new_module_func: function
        :param recursive: indicates this function will replace the modules
                        in the specified pipeline and in all sub pipelines
        :type recursive: bool
        :param force: force replace, skip validation check
        :type force: bool
        :return: pipeline it self
        :rtype: Pipeline
        """
        old_module = old_module_func()
        new_module = new_module_func()
        if not force:
            errors = ModuleValidator.validate_compatibility(old_module, new_module)

            if len(errors) > 0:
                raise UserErrorException('Module incompatible! Errors:{0}'.format(errors))
        self._replace_module(old_module, new_module, recursive)
        return self

    def _expand_pipeline_to_pipelines(self):
        pipelines = []
        _expand_pipeline_to_pipelines(self, pipelines)
        return pipelines

    def _expand_pipeline_nodes(self, prefix="", module_node_to_graph_node_mapping=None):
        """
        Expand pipeline to node list, and mapping of module instance_id to node info

        :param prefix: parent pipeline name
        :type prefix: str
        :param module_node_to_graph_node_mapping: mapping of module node to graph node
        :type module_node_to_graph_node_mapping: dict
        :return: node list and mapping of module instance_id to node info
        :rtype: list, dict({str: dict})
        """
        module_to_node_mapping = {}
        steps = []
        for node in self.nodes:
            if isinstance(node, Module):
                step = node
                setattr(step, 'pipeline', self)
                setattr(step, 'module_node', node)
                module_to_node_mapping[step._instance_id] = {
                    STEP_PREFIX: prefix,
                    NODE_ID:
                        None if not module_node_to_graph_node_mapping
                        else module_node_to_graph_node_mapping[step._instance_id],
                    WORKING_DIR: ''
                }
                steps.append(step)
            elif isinstance(node, Pipeline):
                sub_pipeline_steps, sub_pipeline_module_mapping = \
                    node._expand_pipeline_nodes(os.path.join(prefix, node.name), module_node_to_graph_node_mapping)
                module_to_node_mapping.update(sub_pipeline_module_mapping)
                steps.extend(sub_pipeline_steps)
        return steps, module_to_node_mapping

    def _get_visualization_builder(self):
        module_nodes, _ = self._expand_pipeline_nodes()
        pipelines = self._expand_pipeline_to_pipelines()
        graph_builder_context = _GraphEntityBuilderContext(compute_target=self._get_default_compute_target(),
                                                           module_nodes=module_nodes,
                                                           workspace=self.workspace,
                                                           default_datastore=self.default_datastore)

        graph_entity_builder = _GraphEntityBuilder(graph_builder_context)
        graph, _ = graph_entity_builder.build_graph_entity(is_local_run=True)

        visualization_context = VisualizationContext(pipeline_name=self.name,
                                                     graph=graph,
                                                     module_nodes=module_nodes,
                                                     pipelines=pipelines)
        return VisualizationBuilder(visualization_context)

    def _build_visualization_dict(self):
        visualization_builder = self._get_visualization_builder()
        return visualization_builder.build_visualization_dict()

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Pipeline_run')
    def run(self, experiment_name, working_dir=None, pipeline_parameters=None, show_output=False,
            show_graph=True, continue_on_step_failure=None, max_workers=None):
        """
        Run pipeline in local
        Currently only run basic module in local.

        :param experiment_name: The experiment name
        :type experiment_name: str
        :param working_dir: pipline run data and snapshot store path
        :type working_dir: str
        :param pipeline_parameters: An optional dictionary of pipeline parameter
        :type pipeline_parameters: dict({str:str})
        :param show_output: Indicates whether to show the pipeline run status on sys.stdout.
        :type show_output: bool
        :param show_graph: Indicates whether to show the graph with run status on notebook.
            If not in notebook environment, overwrite this value to False
        :type show_graph: bool
        :param continue_on_step_failure: Indicates whether to continue pipeline execution if a step fails.
            If True, only steps that have no dependency on the output of the failed step will continue execution.
        :type continue_on_step_failure: bool
        :param max_workers:  The maximum number of threads that can be used to execute pipeline steps.
            If max_workers is None, it will default to the number of processors on the machine.
        :type max_workers: int
        :return: pipeline run status
        :rtype: string
        """
        # in notebook show pipeline
        from ._widgets._visualize import _in_jupyter_nb, _visualize
        visualizer = None

        builder = self._get_visualization_builder()
        module_node_to_graph_node_mapping = builder.graph.module_node_to_graph_node_mapping

        if _in_jupyter_nb() and show_graph:
            graphyaml = builder.build_visualization_dict()
            is_prod = _is_prod_workspace(self.workspace)
            envinfo = {
                "subscription_id": self.workspace.subscription_id
            }
            visualizer = _visualize(graphyaml, envinfo=envinfo, is_prod=is_prod)

        # create experiment
        experiment = Experiment(self.workspace, experiment_name)
        run = Run._start_logging(experiment, snapshot_directory=None)

        if not working_dir:
            working_dir = os.path.join(tempfile.gettempdir(), experiment_name, run.id)
        Path(working_dir).mkdir(parents=True, exist_ok=True)

        print('Working dir:', working_dir)
        print('RunId:', run.id)
        print('Link to Azure Machine Learning Portal:', run.get_portal_url())

        pipeline_run_success = True
        try:
            pipeline_run_success = _orchestrate_pipeline_run(self,
                                                             working_dir,
                                                             run,
                                                             module_node_to_graph_node_mapping,
                                                             visualizer=visualizer,
                                                             pipeline_parameters=pipeline_parameters,
                                                             show_output=show_output,
                                                             continue_on_step_failure=continue_on_step_failure,
                                                             max_workers=max_workers)
        except Exception as e:
            run.fail()
            raise e
        if pipeline_run_success:
            run.complete()
        else:
            run.fail()
        return run.get_status()
