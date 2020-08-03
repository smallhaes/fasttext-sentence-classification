# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""service_calller.py, module for interacting with the AzureML service."""

import uuid
import json

from azureml._base_sdk_common import _ClientSessionId
from .designer.designer_service_client import DesignerServiceClient
from msrest.exceptions import HttpOperationError
from .designer.models import SavePipelineDraftRequest, UnpackPackageFromCommunityRequest, \
    PipelineType, PipelineDraftMode, BatchGetModuleRequest, AmlModuleNameMetaInfo
from .pipeline_draft import PipelineDraft
from .._module_dto import ModuleDto
from .._loggerfactory import _LoggerFactory, track
from .._utils import int_str_to_pipeline_status

_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


def _eat_exception_trace(function_name, function, **kwargs):
    result = None
    error_msg = None
    try:
        result = function(**kwargs)
    except HttpOperationError as ex:
        error_msg = "{0} failed with exception: {1}".format(function_name, ex.message)
    return result, error_msg


class DesignerServiceCaller(object):
    """DesignerServiceCaller.
    :param base_url: base url
    :type base_url: Service URL
    :param workspace: workspace
    :type workspace: Workspace
    """

    # The default namespace placeholder is used when namespace is None for get_module API.
    DEFAULT_MODULE_NAMESPACE_PLACEHOLDER = '-'

    def __init__(self, workspace, base_url=None):
        """Initializes DesignerServiceCaller."""
        self._service_context = workspace.service_context
        if base_url is None:
            base_url = self._service_context._get_pipelines_url()
        self._service_endpoint = base_url
        self._caller = DesignerServiceClient(base_url=base_url)
        self._subscription_id = workspace.subscription_id
        self._resource_group_name = workspace.resource_group
        self._workspace_name = workspace.name
        self.auth = workspace._auth_object
        self.cache = {}

    def _get_custom_headers(self):
        custom_header = self.auth.get_authentication_header()
        request_id = str(uuid.uuid4())
        common_header = {
            "x-ms-client-session-id": _ClientSessionId,
            "x-ms-client-request-id": request_id
        }

        custom_header.update(common_header)
        return custom_header

    @track(_get_logger, activity_name="DesignerServiceCaller_submit_pipeline_run")
    def submit_pipeline_run(self, request, node_composition_mode=None):
        """Submit a pipeline run by graph

        :param request:
        :type request: ~designer.models.SubmitPipelineRunRequest
        :param node_composition_mode: Possible values include: 'None',
         'OnlySequential', 'Full'
        :type node_composition_mode: str
        :return: pipeline run id
        :rtype: str
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        result = self._caller.pipeline_runs.submit_pipeline_run(
            body=request,
            node_composition_mode=node_composition_mode,
            subscription_id=self._subscription_id, resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers())

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_submit_pipeline_draft_run")
    def submit_pipeline_draft_run(self, request, draft_id, node_composition_mode=None):
        """Submit a pipelineDraft run

        :param draft_id:
        :type draft_id: str
        :param request:
        :type request: ~designer.models.SubmitPipelineRunRequest
        :param node_composition_mode: Possible values include: 'None',
         'OnlySequential', 'Full'
        :type node_composition_mode: str
        :return: pipeline run id
        :rtype: str
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        result = self._caller.pipeline_drafts.submit_pipeline_run(
            subscription_id=self._subscription_id, resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name, draft_id=draft_id, body=request,
            node_composition_mode=node_composition_mode, custom_headers=self._get_custom_headers())

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_submit_published_pipeline_run")
    def submit_published_pipeline_run(self, request, pipeline_id):
        """Submit a published pipeline run

        :param pipeline_id:
        :type pipeline_id: str
        :param request:
        :type request: ~designer.models.SubmitPipelineRunRequest
        :return: pipeline run id
        :rtype: str
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """

        result = self._caller.published_pipelines.submit_pipeline_run(
            subscription_id=self._subscription_id, resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name, pipeline_id=pipeline_id, body=request,
            custom_headers=self._get_custom_headers())

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_register_module")
    def register_module(self, validate_only=False, module_source_type=None,
                        yaml_file=None, snapshot_source_zip_file=None, devops_artifacts_zip_url=None,
                        anonymous_registration=False, set_as_default=False):
        """Register a module

        :param validate_only:
        :type validate_only: bool
        :param module_source_type:
        :type module_source_type: int
        :param yaml_file:
        :type yaml_file: str
        :param snapshot_source_zip_file:
        :type snapshot_source_zip_file: BinaryIO
        :param devops_artifacts_zip_url:
        :type devops_artifacts_zip_url: str
        :param anonymous_registration:
        :type anonymous_registration: bool
        :param set_as_default:
        :type set_as_default: bool
        :return: ModuleDto
        :rtype: azureml.pipeline.wrapper._module_dto.ModuleDto
        :raises:
         :class:`HttpOperationError<msrest.exceptions.HttpOperationError>`
        """

        properties = json.dumps({
            'ModuleSourceType': module_source_type,
            'YamlFile': yaml_file,
            'DevopsArtifactsZipUrl': devops_artifacts_zip_url,
            'ModuleWorkingMechanism': 'OutputToDataset',
        })

        result = self._caller.module.register_module(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers(),
            validate_only=validate_only,
            properties=properties,
            snapshot_source_zip_file=snapshot_source_zip_file,
            anonymous_registration=anonymous_registration,
            upgrade_if_exists=True,
            set_as_default_version=set_as_default,
            # We must set to False to make sure the module entity only include required parameters.
            # Note that this only affects **params in module entity** but doesn't affect run_setting_parameters.
            include_run_setting_params=False,
        )
        return ModuleDto(result)

    @track(_get_logger, activity_name="DesignerServiceCaller_get_module_snapshot_url_by_id")
    def get_module_snapshot_url_by_id(self, module_id):
        """get module_snapshot_url by id

        :param module_id:
        :type module_id: str
        :return: str
        :rtype: str
        :raises:
         :class:`HttpOperationError<msrest.exceptions.HttpOperationError>`
        """

        result = self._caller.modules.get_module_snapshot_url_by_id(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers(),
            module_id=module_id
        )
        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_create_pipeline_draft")
    def create_pipeline_draft(self, draft_name, draft_description, graph, tags=None, properties=None,
                              module_node_run_settings=None, sub_pipelines_info=None):
        """Create a new pipeline draft with given graph

        :param draft_name:
        :type draft_name: str
        :param draft_description:
        :type draft_description: str
        :param graph:
        :type graph: ~swagger.models.GraphDraftEntity
        :param tags: This is a dictionary
        :type tags: dict[str, str]
        :param properties: This is a dictionary
        :type properties: dict[str, str]
        :param module_node_run_settings: This is run settings for module nodes
        :type module_node_run_settings: List[~swagger.models.GraphModuleNodeRunSetting]
        :param sub_pipelines_info: sub pipelines info for the current graph
        :type sub_pipelines_info: ~swagger.models.SubPipelinesInfo
        :return: str
        :rtype: ~str
        :raises:
         :class:`HttpOperationError`
        """

        request = SavePipelineDraftRequest(
            name=draft_name,
            description=draft_description,
            graph=graph,
            pipeline_type=PipelineType.training_pipeline,  # hard code to Training pipeline
            pipeline_draft_mode=PipelineDraftMode.normal,
            tags=tags,
            properties=properties,
            module_node_run_settings=module_node_run_settings,
            sub_pipelines_info=sub_pipelines_info
        )
        result = self._caller.pipeline_drafts.create_pipeline_draft_extended(
            body=request,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers())

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_save_pipeline_draft")
    def save_pipeline_draft(self, draft_id, draft_name, draft_description, graph, tags=None,
                            module_node_run_settings=None, sub_pipelines_info=None):
        """Save pipeline draft

        :param draft_id:
        :type draft_id: str
        :param draft_name:
        :type draft_name: str
        :param draft_description:
        :type draft_description: str
        :param graph:
        :type graph: ~swagger.models.GraphDraftEntity
        :param tags: This is a dictionary
        :type tags: dict[str, str]
        :param module_node_run_settings: This is run settings for module nodes
        :type module_node_run_settings: List[~swagger.models.GraphModuleNodeRunSetting]
        :param sub_pipelines_info: sub pipelines info for the current graph
        :type sub_pipelines_info: ~swagger.models.SubPipelinesInfo
        """

        request = SavePipelineDraftRequest(
            name=draft_name,
            description=draft_description,
            graph=graph,
            tags=tags,
            module_node_run_settings=module_node_run_settings,
            sub_pipelines_info=sub_pipelines_info
        )
        result = self._caller.pipeline_drafts.save_pipeline_draft(
            draft_id=draft_id,
            body=request,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers())

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_draft")
    def get_pipeline_draft(self, draft_id, get_status=True, include_run_setting_params=True):
        """Get pipeline draft

        :param draft_id:
        :type draft_id: str
        :return: PipelineDraft
        :rtype: PipelineDraft
        :raises:
         :class:`HttpOperationError`
        """

        result = self._caller.pipeline_drafts.get_pipeline_draft(
            draft_id=draft_id,
            get_status=get_status,
            include_run_setting_params=include_run_setting_params,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers())

        return PipelineDraft(
            raw_pipeline_draft=result,
            subscription_id=self._subscription_id,
            resource_group=self._resource_group_name,
            workspace_name=self._workspace_name)

    @track(_get_logger, activity_name="DesignerServiceCaller_delete_pipeline_draft")
    def delete_pipeline_draft(self, draft_id):
        """Delete pipeline draft

        :param draft_id:
        :type draft_id: str
        :return: PipelineDraft
        :rtype: ~swagger.models.PipelineDraft
        :raises:
         :class:`HttpOperationError`
        """

        result = self._caller.pipeline_drafts.delete_pipeline_draft(
            draft_id=draft_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers())

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_publish_pipeline_run")
    def publish_pipeline_run(self, request, pipeline_run_id):
        """
        Publish pipeline run by pipeline run id

        :param request:
        :type ~designer.models.CreatePublishedPipelineRequest
        :param pipeline_run_id: pipeline_run_id
        :type pipeline_run_id: str
        :return: str or ClientRawResponse if raw=true
        :rtype: str or ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        result = self._caller.pipeline_runs.publish_pipeline_run(
            body=request,
            pipeline_run_id=pipeline_run_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_publish_pipeline_graph")
    def publish_pipeline_graph(self, request):
        """
        Publish pipeline run by pipeline run id

        :param request:
        :type ~designer.models.CreatePublishedPipelineRequest
        :return: str or ClientRawResponse if raw=true
        :rtype: str or ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        result = self._caller.published_pipelines.publish_pipeline_graph(
            body=request,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_list_published_pipelines")
    def list_published_pipelines(self, workspace):
        """
        List all published pipelines in workspace

        :param workspace: The workspace the published pipeline was created in.
        :type workspace: azureml.core.Workspace
        :return: list[PublishedPipeline]
        :rtype: List[azureml.pipeline.wrapper._published_pipeline.PublishedPipeline]
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        continuation_token = None
        results = []
        while True:
            paginated_results = self._caller.published_pipelines.list_published_pipelines(
                subscription_id=self._subscription_id,
                resource_group_name=self._resource_group_name,
                workspace_name=self._workspace_name,
                custom_headers=self._get_custom_headers(),
                continuation_token=continuation_token
            )
            continuation_token = paginated_results.continuation_token
            results += paginated_results.value
            if continuation_token is None:
                break

        from .._published_pipeline import PublishedPipeline
        return [PublishedPipeline(
            id=result.id, name=result.name, description=result.description, graph_id=result.graph_id,
            published_date=result.published_date, last_run_time=result.last_run_time,
            last_run_status=result.last_run_status, published_by=result.published_by, tags=result.tags,
            entity_status=int_str_to_pipeline_status(result.entity_status), created_date=result.created_date,
            last_modified_date=result.last_modified_date, workspace=workspace) for result in results]

    @track(_get_logger, activity_name="DesignerServiceCaller_get_published_pipeline")
    def get_published_pipeline(self, workspace, pipeline_id):
        """
        Get published pipeline by pipeline id

        :param workspace: The workspace the published pipeline was created in.
        :type workspace: azureml.core.Workspace
        :param pipeline_id: pipeline_id
        :type pipeline_id: str
        :return: PublishedPipeline
        :rtype: azureml.pipeline.wrapper._published_pipeline.PublishedPipeline
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        result = self._caller.published_pipelines.get_published_pipeline(
            pipeline_id=pipeline_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )
        from .._published_pipeline import PublishedPipeline
        return PublishedPipeline(
            id=result.id, name=result.name, description=result.description, total_run_steps=result.total_run_steps,
            total_runs=result.total_runs, parameters=result.parameters, rest_endpoint=result.rest_endpoint,
            graph_id=result.graph_id, published_date=result.published_date, last_run_time=result.last_run_time,
            last_run_status=result.last_run_status, published_by=result.published_by, tags=result.tags,
            entity_status=int_str_to_pipeline_status(result.entity_status), created_date=result.created_date,
            last_modified_date=result.last_modified_date, workspace=workspace)

    @track(_get_logger, activity_name="DesignerServiceCaller_enable_published_pipeline")
    def enable_published_pipeline(self, pipeline_id):
        """
        Enable published pipeline by pipeline id

        :param pipeline_id: pipeline_id
        :type pipeline_id: str
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        self._caller.published_pipelines.enable_published_pipeline(
            pipeline_id=pipeline_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

    @track(_get_logger, activity_name="DesignerServiceCaller_disable_published_pipeline")
    def disable_published_pipeline(self, pipeline_id):
        """
        Disable published pipeline by pipeline id

        :param pipeline_id: pipeline_id
        :type pipeline_id: str
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        self._caller.published_pipelines.disable_published_pipeline(
            pipeline_id=pipeline_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_endpoint")
    def get_pipeline_endpoint(self, pipeline_endpoint_id):
        """
        Get pipeline endpoint by id

        :param pipeline_endpoint_id: pipeline_endpoint_id
        :type pipeline_endpoint_id: str
        :return: PipelineEndpoint or ClientRawResponse if raw=true
        :rtype: ~designer.models.PipelineEndpoint or
         ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        result = self._caller.pipeline_endpoints.get_pipeline_endpoint(
            pipeline_endpoint_id=pipeline_endpoint_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipelines_from_pipeline_endpoint")
    def get_pipelines_from_pipeline_endpoint(self, pipeline_endpoint_id, continuation_token=None):
        """
        Get pipeline endpoint by id

        :param pipeline_endpoint_id: pipeline_endpoint_id
        :type pipeline_endpoint_id: str
        :return: PaginatedPublishedPipelineSummaryList or ClientRawResponse if
         raw=true
        :rtype: ~designer.models.PaginatedPublishedPipelineSummaryList or
         ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        result = self._caller.pipeline_endpoints.get_pipeline_endpoint_pipelines(
            pipeline_endpoint_id=pipeline_endpoint_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers(),
            continuation_token=continuation_token
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_list_pipeline_endpoint")
    def list_pipeline_endpoint(self, continuation_token=None):
        """
        list pipeline endpoint

        :return: PaginatedPipelineEndpointSummaryList or ClientRawResponse if
         raw=true
        :rtype: ~designer.models.PaginatedPipelineEndpointSummaryList or
         ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """
        result = self._caller.pipeline_endpoints.list_pipeline_endpoints(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers(),
            continuation_token=continuation_token
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_delete_pipeline_draft")
    def list_pipeline_drafts(self, continuation_token=None):
        """List pipeline draft

        :param draft_id:
        :type draft_id: str
        :return: PipelineDraft
        :rtype: ~swagger.models.PipelineDraft
        :raises:
         :class:`ErrorResponseException`
        """

        result = self._caller.pipeline_drafts.list_pipeline_drafts(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            continuation_token1=continuation_token,
            custom_headers=self._get_custom_headers())

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_list_samples")
    def list_samples(self):
        """List all of our samples
        """

        self._caller.packages.list_samples.metadata['url'] = \
            '/studioservice/api/' \
            'subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/workspaces/{workspaceName}/' \
            'Packages/samples?$skip=0&$orderby=trending%20desc&$filter=author%2Fid%20eq%20%' \
            '272ACABB3CD54D7C702886335B49C9ED30A5E9447D0918DB6DE242AB88908C4178%27'
        result = self._caller.packages.list_samples(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers(),
            raw=True)

        resp_body = json.loads(result.response.content)
        sample_list = [{'name': sample['name'], 'id': sample['id']}
                       for sample in resp_body['value']]

        return sample_list

    @track(_get_logger, activity_name="DesignerServiceCaller_open_sample")
    def open_sample(self, sample_id):
        """Open sample by sample id
        """

        request = UnpackPackageFromCommunityRequest(
            community_item_id=sample_id,
            pipeline_type=PipelineType.training_pipeline,
            pipeline_draft_mode=PipelineDraftMode.normal
        )
        result = self._caller.packages.unpack_and_get_draft_from_comunity(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers(),
            body=request
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_list_datasets")
    def list_datasets(self, data_category="0"):
        """List datasets by category

        :param data_category: Possible values include: 'All', 'Dataset',
         'Model'
        :type data_category: str
        :return: list
        :rtype: list[~designer.models.DataInfo]
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        result = self._caller.data_sets.list_data_sets(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers(),
            data_category=data_category
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_run_graph")
    def get_pipeline_run_graph(self, pipeline_run_id, experiment_name=None, experiment_id=None,
                               include_run_setting_params=True):
        """Get pipeline run graph

        :param pipeline_run_id:
        :type pipeline_run_id: str
        :param experiment_name:
        :type experiment_name: str
        :param experiment_id:
        :type experiment_id: str
        :param include_run_setting_params:
        :type include_run_setting_params: bool
        :return: PipelineRunGraphDetail
        :rtype: ~designer.models.PipelineRunGraphDetail
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        result = self._caller.pipeline_runs.get_pipeline_run_graph(
            pipeline_run_id=pipeline_run_id,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            include_run_setting_params=include_run_setting_params,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_run_graph_no_status")
    def get_pipeline_run_graph_no_status(self, pipeline_run_id, include_run_setting_params=True):
        """Get pipeline run graph no status

        :param pipeline_run_id:
        :type pipeline_run_id: str
        :param include_run_setting_params:
        :type include_run_setting_params: bool
        :return: PipelineRunGraphDetail
        :rtype: ~designer.models.PipelineRunGraphDetail
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        result = self._caller.pipeline_runs.get_pipeline_run_graph_no_status(
            pipeline_run_id=pipeline_run_id,
            include_run_setting_params=include_run_setting_params,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_draft_sdk_code")
    def get_pipeline_draft_sdk_code(self, draft_id, target_code):
        """Export pipeline draft to sdk code

        :param draft_id: the draft to export
        :type draft_id: str
        :param target_code: specify the exported code type: Python or JupyterNotebook
        :type target_code: str
        :return: str or ClientRawResponse if raw=true
        :rtype: str or ~msrest.pipeline.ClientRawResponse
        :raises:
        :class:`HttpOperationError<designer.models.HttpOperationError>`
        """
        result = self._caller.pipeline_drafts.get_pipeline_draft_sdk_code(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            draft_id=draft_id,
            target_code=target_code,
            raw=True,
            custom_headers=self._get_custom_headers()
        )

        return result.response

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_run_sdk_code")
    def get_pipeline_run_sdk_code(self, pipeline_run_id, target_code, experiment_name, experiment_id):
        """Export pipeline run to sdk code

        :param pipeline_run_id: the pipeline run to export
        :type pipeline_run_id: str
        :param target_code: specify the exported code type: Python or JupyterNotebook
        :type target_code: str
        :param experiment_name: the experiment that contains the run
        :type experiment_name: str
        :param experiment_id: the experiment that contains the run
        :type experiment_id: str
        :return: str or ClientRawResponse if raw=true
        :rtype: str or ~msrest.pipeline.ClientRawResponse
        :raises:
        :class:`HttpOperationError<designer.models.HttpOperationError>`
        """
        result = self._caller.pipeline_runs.get_pipeline_run_sdk_code(
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            pipeline_run_id=pipeline_run_id,
            target_code=target_code,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            raw=True,
            custom_headers=self._get_custom_headers()
        )

        return result.response

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_run_status")
    def get_pipeline_run_status(self, pipeline_run_id, experiment_name=None, experiment_id=None):
        """Get pipeline run status

        :param pipeline_run_id:
        :type pipeline_run_id: str
        :param experiment_name:
        :type experiment_name: str
        :param experiment_id:
        :type experiment_id: str
        :return: PipelineRunGraphStatus
        :rtype: ~designer.models.PipelineRunGraphStatus
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        result = self._caller.pipeline_runs.get_pipeline_run_status(
            pipeline_run_id=pipeline_run_id,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_get_module_versions")
    def get_module_versions(self, module_namespace, module_name):
        """Get module dtos

        :param module_namespace:
        :type module_namespace: str
        :param module_name:
        :type module_name: str
        :return: dict
        :rtype: dict[str, azureml.pipeline.wrapper._module_dto.ModuleDto]
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        result = self._caller.module.get_module_versions(
            module_namespace=module_namespace,
            module_name=module_name,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        result = {k: ModuleDto(v) for k, v in result}

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_batch_get_modules")
    def batch_get_modules(self, module_version_ids, name_identifiers, include_run_setting_params=True):
        """Get modules dto

        :param module_version_ids:
        :type module_version_ids: list
        :param name_identifiers:
        :type name_identifiers: list
        :return: modules_by_id concat modules_by_identifier
        :rtype: List[azureml.pipeline.wrapper._module_dto.ModuleDto]
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """
        aml_modules = [AmlModuleNameMetaInfo(module_name=item[0],
                                             module_namespace=item[1],
                                             module_version=item[2]) for item in name_identifiers]
        request = BatchGetModuleRequest(
            module_version_ids=module_version_ids,
            aml_modules=aml_modules
        )
        result, error_msg = \
            _eat_exception_trace("Batch load modules",
                                 self._caller.modules.batch_get_modules,
                                 body=request,
                                 include_run_setting_params=include_run_setting_params,
                                 subscription_id=self._subscription_id,
                                 resource_group_name=self._resource_group_name,
                                 workspace_name=self._workspace_name,
                                 custom_headers=self._get_custom_headers())
        if result is not None:
            result = [ModuleDto(item) for item in result]

        return result, error_msg

    @track(_get_logger, activity_name="DesignerServiceCaller_get_module")
    def get_module(self, module_namespace, module_name, version=None, include_run_setting_params=True):
        """Get module dto

        :param module_namespace:
        :type module_namespace: str
        :param module_name:
        :type module_name: str
        :param version:
        :type version: str
        :param include_run_setting_params:
        :type include_run_setting_params: bool
        :return: ModuleDto
        :rtype: azureml.pipeline.wrapper._module_dto.ModuleDto
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        # Set the default placeholder of module namespace when it is None.
        module_namespace = module_namespace if module_namespace else self.DEFAULT_MODULE_NAMESPACE_PLACEHOLDER
        result = self._caller.module.get_module(
            module_namespace=module_namespace,
            module_name=module_name,
            version=version,
            include_run_setting_params=include_run_setting_params,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return ModuleDto(result)

    @track(_get_logger, activity_name="DesignerServiceCaller_get_module_by_id")
    def get_module_by_id(self, module_id, include_run_setting_params=True):
        """Get module dto by module id
        """

        result = self._caller.modules.get_module_dto_by_id(
            module_id=module_id,
            include_run_setting_params=include_run_setting_params,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return ModuleDto(result)

    @track(_get_logger, activity_name="DesignerServiceCaller_get_module_yaml_by_id")
    def get_module_yaml_by_id(self, module_id):
        """Get module yaml by module id
        """

        result = self._caller.modules.get_module_yaml_by_id(
            module_id=module_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers(),
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_run_step_details")
    def get_pipeline_run_step_details(self, pipeline_run_id, run_id, include_snaptshot=False):
        """Get pipeline step run details

        :param pipeline_run_id:
        :type pipeline_run_id: str
        :param run_id:
        :type run_id: str
        :param include_snaptshot:
        :type include_snaptshot: bool
        :param dict custom_headers: headers that will be added to the request
        :param bool raw: returns the direct response alongside the
         deserialized response
        :param operation_config: :ref:`Operation configuration
         overrides<msrest:optionsforoperations>`.
        :return: PipelineRunStepDetails or ClientRawResponse if raw=true
        :rtype: ~designer.models.PipelineRunStepDetails or
         ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`HttpOperationError<designer.models.HttpOperationError>`
        """

        result = self._caller.pipeline_runs.get_pipeline_run_step_details(
            pipeline_run_id=pipeline_run_id,
            run_id=run_id,
            include_snaptshot=include_snaptshot,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_get_pipeline_run")
    def get_pipeline_run(self, pipeline_run_id):
        """

        :param pipeline_run_id:
        :type pipeline_run_id: str
        :return: PipelineRun
        :rtype: ~designer.models.PipelineRun or
         ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """

        result = self._caller.pipeline_runs.get_pipeline_run(
            pipeline_run_id=pipeline_run_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )

        return result

    @track(_get_logger, activity_name="DesignerServiceCaller_list_experiment_computes")
    def list_experiment_computes(self, include_test_types=False, use_sdk_cache=True):
        """

        :type include_test_types: bool
        :param dict custom_headers: headers that will be added to the request
        :return: list
        :rtype: list[~designer.models.ExperimentComputeMetaInfo]
        :raises:
         :class:`ErrorResponseException<designer.models.ErrorResponseException>`
        """

        cache_key = f'list_experiment_computes?include_test_types={include_test_types}'
        if use_sdk_cache and cache_key in self.cache:
            return self.cache[cache_key]

        result = self._caller.computes.list_experiment_computes(
            include_test_types=include_test_types,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            workspace_name=self._workspace_name,
            custom_headers=self._get_custom_headers()
        )
        # Add to cache
        self.cache[cache_key] = result

        return result
