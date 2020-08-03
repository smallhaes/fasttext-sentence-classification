# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Tuple, Callable

from azureml.core import Workspace, Experiment, Run
from .pipeline_run import PipelineRun
from ._published_pipeline import PublishedPipeline
from ._restclients.service_caller import DesignerServiceCaller
from ._restclients.designer.models import SubmitPipelineRunRequest, GraphDraftEntity, \
    CreatePublishedPipelineRequest


class PipelineProviderContext(object):
    def __init__(self, experiment_name: str = None, graph=None, sub_pipelines_info=None, module_node_run_settings=None,
                 pipeline_parameters=None, continue_on_step_failure: bool = None, submit_description=None,
                 regenerate_outputs: bool = None, compute_target=None, tags=None, properties=None):
        """
        The context needed for pipeline in backend.

        :param experiment_name: The experiment name
        :type experiment_name: str
        :param graph: The graph entity.
        :type graph: GraphDraftEntity.
        :param sub_pipelines_info: Sub pipelines related info.
        :type sub_pipelines_info: dict
        :param module_node_run_settings: Module node runsettings.
        :type module_node_run_settings: list
        :param pipeline_parameters: An optional dictionary of pipeline parameter assignments for the PipelineDraft
        :type pipeline_parameters: dict({str:str})
        :param continue_on_step_failure: start pipeline run from last failure position
        :type continue_on_step_failure: bool
        :param regenerate_outputs: disable reuse output for all nodes in pipeline
        :type regenerate_outputs: bool
        :param submit_description: description of the submitted pipeline run
        :type submit_description: str
        :param compute_target: The compute target used to run pipeline
        :type compute_target: str
        :param tags: Tags to be added to the submitted run, {"tag": "value"}
        :type tags: dict
        :param properties: Optional properties dictionary for the PipelineDraft,
            only needed when saving as a new PipelineDraft
        :type properties: dict({str:str})
        """
        self.experiment_name = experiment_name
        self.graph = graph
        self.sub_pipelines_info = sub_pipelines_info
        self.module_node_run_settings = module_node_run_settings
        self.pipeline_parameters = pipeline_parameters
        self.continue_on_step_failure = continue_on_step_failure
        self.submit_description = submit_description
        self.regenerate_outputs = regenerate_outputs
        self.compute_target = compute_target
        self.tags = tags
        self.properties = properties


class PublishedPipelineProviderContext(PipelineProviderContext):
    def __init__(self, pipeline_name: str = None, pipeline_description: str = None, pipeline_endpoint_name: str = None,
                 pipeline_endpoint_description: str = None, set_as_default: bool = True,
                 use_existing_pipeline_endpoint: bool = False, use_pipeline_endpoint: bool = False,
                 pipeline_id: str = False, run_id: str = None, **kwargs):
        """
        The context needed for published pipeline in backend.

        :param pipeline_name: The name of the published pipeline.
        :type pipeline_name: str
        :param pipeline_description: The description of the published pipeline.
        :type pipeline_description: str
        :param pipeline_endpoint_name: The name of pipeline endpoint.
        :type pipeline_endpoint_name: str
        :param pipeline_endpoint_description: The description of pipeline endpoint.
        :type pipeline_endpoint_description: str
        :param set_as_default: Whether to use pipeline published as the default version of pipeline endpoint.
        :type set_as_default: bool
        :param use_existing_pipeline_endpoint: Whether to use existing pipeline endpoint.
        :type use_existing_pipeline_endpoint: bool
        :param use_pipeline_endpoint: use pipeline endpoint to manage published_pipeline or not.
        :type use_pipeline_endpoint: bool
        :param pipeline_id: the id of published pipeline when submit
        :type pipeline_id: str
        :param run_id: the id of pipeline run to publish
        :type run_id: str
        """
        super().__init__(**kwargs)
        self.pipeline_name = pipeline_name
        self.pipeline_description = pipeline_description
        self.pipeline_endpoint_name = pipeline_endpoint_name
        self.pipeline_endpoint_description = pipeline_endpoint_description
        self.set_as_default = set_as_default
        self.use_existing_pipeline_endpoint = use_existing_pipeline_endpoint
        self.use_pipeline_endpoint = use_pipeline_endpoint
        self.pipeline_id = pipeline_id
        self.run_id = run_id


class AbstractBackendProvider(ABC):
    """Abstract base class for backend provider.

    The derived class should provides backend service implementation for pipeline.
    """

    @abstractmethod
    def submit_pipeline(self, workspace: Workspace,
                        pipeline,
                        context: PipelineProviderContext) -> Tuple[Run, GraphDraftEntity]:
        """ Submit pipeline run to workspace.

        :param workspace: The workspace
        :type workspace: azureml.core.Workspace
        :param pipeline: The pipeline to submit
        :type pipeline: azureml.pipeline.wrapper.Pipeline
        :param context: The context needed.
        :type context: PipelineProviderContext
        :return: run
        :rtype: azureml.core.Run
        """
        raise NotImplementedError

    @abstractmethod
    def save_pipeline_as_draft(self, _id, workspace: Workspace,
                               pipeline,
                               context: PipelineProviderContext):
        """Save pipeline as PipelineDraft.

        :param _id: Existing pipeline draft id. If specified, pipeline will be save to that pipeline draft.
        :type _id: str
        :param workspace: The workspace
        :type workspace: azureml.core.Workspace
        :param pipeline: The pipeline to submit
        :type pipeline: azureml.pipeline.wrapper.Pipeline
        :param context: The context needed.
        :type context: PipelineProviderContext

        :return: The created PipelineDraft.
        :rtype: azureml.pipeline.core.PipelineDraft
        """
        raise NotImplementedError

    @abstractmethod
    def get_pipeline_run_status(self, experiment: Experiment, run_id: str):
        """Retrieve pipeline run status.

        :param experiment: The experiment object associated with the pipeline run.
        :type experiment: azureml.core.Experiment
        :param run_id: The run ID of the pipeline run.
        :type run_id: str
        :param run_id:
        :return: run status
        """
        raise NotImplementedError


class SmtBackendProvider(AbstractBackendProvider):

    def submit_pipeline(self, workspace: Workspace,
                        pipeline,
                        context: PipelineProviderContext) -> Run:
        service_caller = DesignerServiceCaller(workspace)
        compute_target_name, _ = context.compute_target
        request = SubmitPipelineRunRequest(
            experiment_name=context.experiment_name,
            description=context.submit_description if context.submit_description is not None else pipeline.name,
            compute_target=compute_target_name,
            graph=context.graph,
            module_node_run_settings=context.module_node_run_settings,
            tags=context.tags,
            continue_run_on_step_failure=context.continue_on_step_failure,
            sub_pipelines_info=context.sub_pipelines_info
        )

        # Special case for kubeflow
        draft_id = None
        if compute_target_name is not None and "kubeflow" in compute_target_name:
            draft = self.save_pipeline_as_draft(_id=None, workspace=workspace,
                                                pipeline=pipeline, context=context)
            draft_id = draft.id
            run_id = service_caller.submit_pipeline_draft_run(request=request, draft_id=draft_id)
        else:
            run_id = service_caller.submit_pipeline_run(request)

        print('Submitted PipelineRun', run_id)
        experiment = Experiment(workspace, context.experiment_name)
        run = PipelineRun(experiment, run_id)
        print('Link to Azure Machine Learning Portal:', run.get_portal_url())
        return run

    def submit_published_pipeline(self, workspace: Workspace, context) -> Run:
        service_caller = DesignerServiceCaller(workspace)
        request = SubmitPipelineRunRequest(
            experiment_name=context.experiment_name,
            description=context.pipeline_description,
            pipeline_parameters=context.pipeline_parameters
        )

        run_id = service_caller.submit_published_pipeline_run(request=request, pipeline_id=context.pipeline_id)
        print('Submitted PublishedPipelineRun', run_id)
        experiment = Experiment(workspace, context.experiment_name)
        run = PipelineRun(experiment, run_id)
        print('Link to Azure Machine Learning Portal:', run.get_portal_url())
        return run

    def save_pipeline_as_draft(self, _id, workspace: Workspace,
                               pipeline,
                               context: PipelineProviderContext):
        service_caller = DesignerServiceCaller(workspace)
        if _id is None:
            pipeline_draft_id = service_caller.create_pipeline_draft(
                draft_name=pipeline.name,
                draft_description=pipeline.description,
                graph=context.graph,
                module_node_run_settings=context.module_node_run_settings,
                tags=context.tags,
                properties=context.properties,
                sub_pipelines_info=context.sub_pipelines_info)
            pipeline_draft = service_caller.get_pipeline_draft(
                pipeline_draft_id, include_run_setting_params=False)
        else:
            service_caller.save_pipeline_draft(
                draft_id=_id,
                draft_name=pipeline.name,
                draft_description=pipeline.description,
                graph=context.graph,
                sub_pipelines_info=context.sub_pipelines_info,
                module_node_run_settings=context.module_node_run_settings,
                tags=context.tags)
            pipeline_draft = service_caller.get_pipeline_draft(
                _id, include_run_setting_params=False)
        return pipeline_draft

    def publish_pipeline(self, workspace: Workspace, context, pipeline=None) -> PublishedPipeline:
        if context.run_id is None and pipeline is None:
            raise Exception('Both "run_id" and "pipeline" not specified.')

        # Assert pipeline endpoint has unique name
        service_caller = DesignerServiceCaller(workspace)

        def find_pipeline_endpoint_by_name(_endpoint, _name):
            return _endpoint.name == _name

        if context.use_pipeline_endpoint:
            endpoint = self.get_pipeline_endpoint_by_func(workspace=workspace,
                                                          func=find_pipeline_endpoint_by_name,
                                                          _name=context.pipeline_endpoint_name)
            if endpoint is None:
                print('Creating a new pipeline endpoint name "{0}"'.format(context.pipeline_endpoint_name))
                context.use_existing_pipeline_endpoint = False
            else:
                if context.use_existing_pipeline_endpoint is False:
                    raise Exception('Pipeline endpoint "{0}" already exists!'.format(
                        context.pipeline_endpoint_name))
                else:
                    print('Using existing pipeline endpoint "{0}"'.format(context.pipeline_endpoint_name))

        request = CreatePublishedPipelineRequest(
            pipeline_name=context.pipeline_name,
            experiment_name=context.experiment_name,
            pipeline_description=context.pipeline_description,
            pipeline_endpoint_name=context.pipeline_endpoint_name,
            pipeline_endpoint_description=context.pipeline_endpoint_description,
            tags=context.tags,
            graph=context.graph,
            set_as_default_pipeline_for_endpoint=context.set_as_default,
            use_existing_pipeline_endpoint=context.use_existing_pipeline_endpoint,
            use_pipeline_endpoint=context.use_pipeline_endpoint,
            properties=context.properties
        )

        result_id = service_caller.publish_pipeline_run(
            request=request,
            pipeline_run_id=context.run_id
        ) if context.run_id is not None \
            else service_caller.publish_pipeline_graph(
            request=request
        )
        return service_caller.get_published_pipeline(workspace=workspace, pipeline_id=result_id)

    def get_pipeline_endpoint_by_func(self, workspace: Workspace, func: Callable, **kwargs):
        """
        Get pipeline endpoint by a function
        """
        service_caller = DesignerServiceCaller(workspace)
        continuation_token = None
        while True:
            paginated_endpoints = service_caller.list_pipeline_endpoint(continuation_token=continuation_token)
            for endpoint in paginated_endpoints.value:
                if func(endpoint, **kwargs):
                    return service_caller.get_pipeline_endpoint(endpoint.id)
            continuation_token = paginated_endpoints.continuation_token
            if continuation_token is None:
                return None

    def get_pipeline_run_status(self, experiment: Experiment, run_id: str):
        service_caller = DesignerServiceCaller(workspace=experiment.workspace)
        result = service_caller.get_pipeline_run_status(pipeline_run_id=run_id, experiment_name=experiment.name,
                                                        experiment_id=experiment.id)
        return result
