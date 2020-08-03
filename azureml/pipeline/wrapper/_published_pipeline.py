# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml.core import Workspace

from ._restclients.service_caller import DesignerServiceCaller
from ._telemetry import _get_telemetry_value_from_workspace, _get_telemetry_value_from_pipeline_parameter
from ._loggerfactory import _LoggerFactory, _PUBLIC_API, track

_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


class PublishedPipeline(object):
    """
    PublishedPipeline

    :param id: The ID of the published pipeline.
    :type id: str
    :param name: The name of the published pipeline.
    :type name: str
    :param description: The description of the published pipeline.
    :type description: str
    :param total_run_steps: The number of steps in this pipeline.
    :type total_run_steps: int
    :param total_runs: The number of runs in this pipeline.
    :type total_runs: int
    :param parameters: parameters of published pipeline.
    :type parameters: dict[str, str]
    :param rest_endpoint: The REST endpoint URL to submit runs for this pipeline.
    :type rest_endpoint: str
    :param graph_id: The ID of the graph for this published pipeline.
    :type graph_id: str
    :param published_date: The published date of this pipeline.
    :type published_date: datetime
    :param last_run_time: The last run time of this pipeline.
    :type last_run_time: datetime
    :param last_run_status: Possible values include: 'NotStarted', 'Running',
     'Failed', 'Finished', 'Canceled'.
    :type last_run_status: str or ~designer.models.PipelineRunStatusCode
    :param published_by: user name who published this pipeline.
    :type published_by: str
    :param tags: tags of pipeline.
    :type tags: dict[str, str]
    :param is_default: if pipeline is the default one in pipeline_endpoint.
    :type is_default: bool
    :param entity_status: Possible values include: 'Active', 'Deprecated',
     'Disabled'
    :type entity_status: str or ~designer.models.EntityStatus
    :param created_date: create date of pipeline.
    :type created_date: datetime
    :param last_modified_date: last modified date of published pipeline.
    :type last_modified_date: datetime
    :param workspace: The workspace of the published pipeline.
    :type workspace: azureml.core.Workspace
    """

    def __init__(self, id: str = None, name: str = None, description: str = None, total_run_steps: int = None,
                 total_runs: int = None, parameters: dict = None, rest_endpoint: str = None, graph_id: str = None,
                 published_date=None, last_run_time=None, last_run_status=None, published_by: str = None,
                 tags=None, entity_status=None, created_date=None, last_modified_date=None,
                 workspace: Workspace = None) -> None:
        """
        Initialize PublishedPipeline

        :param id: The ID of the published pipeline.
        :type id: str
        :param name: The name of the published pipeline.
        :type name: str
        :param description: The description of the published pipeline.
        :type description: str
        :param total_run_steps: The number of steps in this pipeline.
        :type total_run_steps: int
        :param total_runs: The number of runs in this pipeline.
        :type total_runs: int
        :param parameters: parameters of published pipeline.
        :type parameters: dict[str, str]
        :param rest_endpoint: The REST endpoint URL to submit runs for this pipeline.
        :type rest_endpoint: str
        :param graph_id: The ID of the graph for this published pipeline.
        :type graph_id: str
        :param published_date: The published date of this pipeline.
        :type published_date: datetime
        :param last_run_time: The last run time of this pipeline.
        :type last_run_time: datetime
        :param last_run_status: Possible values include: 'NotStarted', 'Running',
         'Failed', 'Finished', 'Canceled'.
        :type last_run_status: str or ~designer.models.PipelineRunStatusCode
        :param published_by: user name who published this pipeline.
        :type published_by: str
        :param tags: tags of pipeline.
        :type tags: dict[str, str]
        :param entity_status: Possible values include: 'Active', 'Deprecated',
         'Disabled'
        :type entity_status: str or ~designer.models.EntityStatus
        :param created_date: create date of pipeline.
        :type created_date: datetime
        :param last_modified_date: last modified date of published pipeline.
        :type last_modified_date: datetime
        :param workspace: The workspace of the published pipeline.
        :type workspace: azureml.core.Workspace
        """
        self._id = id
        self._name = name
        self._description = description
        self._total_run_steps = total_run_steps
        self._total_runs = total_runs
        self._parameters = parameters
        self._rest_endpoint = rest_endpoint
        self._graph_id = graph_id
        self._published_date = published_date
        self._last_run_time = last_run_time
        self._last_run_status = last_run_status
        self._published_by = published_by
        self._tags = tags
        self._status = entity_status
        self._created_date = created_date
        self._last_modified_date = last_modified_date
        self._workspace = workspace
        from ._backend_provider import SmtBackendProvider
        self._backend_provider = SmtBackendProvider()

    @property
    def id(self):
        """
        Property method to get published_pipeline's id.

        :return: The id.
        :rtype: str
        """
        return self._id

    @property
    def name(self):
        """
        Property method to get published_pipeline's name.

        :return: The name.
        :rtype: str
        """
        return self._name

    @property
    def description(self):
        """
        Property method to get published_pipeline's description.

        :return: The description.
        :rtype: str
        """
        return self._description

    @property
    def workspace(self):
        """
        Property method to get published_pipeline's workspace

        :return: The workspace.
        :rtype: azureml.core.Workspace
        """
        return self._workspace

    @property
    def rest_endpoint(self):
        """
        Property method to get published_pipeline's rest_endpoint url.

        :return: The rest_endpoint.
        :rtype: str
        """
        return self._rest_endpoint

    @property
    def status(self):
        """
        Property method to get published_pipeline's status.

        :return: The status. Possible values include: 'Active', 'Deprecated',
         'Disabled'.
        :rtype: str
        """
        return self._status

    @staticmethod
    def get(workspace, id):
        """
        Get the published pipeline.

        :param workspace: The workspace the published pipeline was created in.
        :type workspace: azureml.core.Workspace
        :param id: The ID of the published pipeline.
        :type id: str

        :return: A PublishedPipeline object.
        :rtype: azureml.pipeline.wrapper._published_pipeline.PublishedPipeline
        """
        service_caller = DesignerServiceCaller(workspace)
        telemetry_values = _get_telemetry_value_from_workspace(workspace)
        telemetry_values.update({'pipeline_id': id})
        _LoggerFactory.trace(_get_logger(), "PublishedPipeline_get", telemetry_values)
        return service_caller.get_published_pipeline(workspace, pipeline_id=id)

    @staticmethod
    def list(workspace):
        """
        Get all (includes disabled) published pipelines which has no related pipeline endpoint
            in the current workspace.
        None of returned published pipeline have `total_run_steps`, `total_runs`,
            `parameters` and `rest_endpoint` attribute, you can get them by `PublishedPipeline.get()`
            function with workspace and published pipeline id.

        :param workspace: The workspace the published pipeline was created in.
        :type workspace: azureml.core.Workspace

        :return: A list of PublishedPipeline objects.
        :rtype: builtin.list[azureml.pipeline.wrapper._published_pipeline.PublishedPipeline]
        """

        service_caller = DesignerServiceCaller(workspace)
        telemetry_values = _get_telemetry_value_from_workspace(workspace)
        _LoggerFactory.trace(_get_logger(), "PublishedPipeline_list", telemetry_values)
        return service_caller.list_published_pipelines(workspace)

    def enable(self):
        """Set the published pipeline to 'Active' and available to run."""
        service_caller = DesignerServiceCaller(self.workspace)
        service_caller.enable_published_pipeline(self._id)
        self._status = 'Active'

    def disable(self):
        """Set the published pipeline to 'Disabled' and unavailable to run."""
        service_caller = DesignerServiceCaller(self.workspace)
        service_caller.disable_published_pipeline(self._id)
        self._status = 'Disabled'

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='PublishedPipeline_submit')
    def submit(self, experiment_name: str, description: str = None, parameters: dict = None):
        """
        Submit the published pipeline.

        Returns the submitted :class:`azureml.pipeline.wrapper.PipelineRun`. Use this object to monitor and
        view details of the run.
        PipelineEndpoints can be used to create new versions of a :class:`azureml.pipeline.core.PublishedPipeline`
        while maintaining the same endpoint. PipelineEndpoints are uniquely named within a workspace.

        Using the endpoint attribute of a PipelineEndpoint object, you can trigger new pipeline runs from external
        applications with REST calls. For information about how to authenticate when calling REST endpoints, see
        https://aka.ms/pl-restep-auth.

        :param experiment_name: The name of the experiment to submit to.
        :type experiment_name: str
        :param description: A clear description to distinguish runs.
        :type description: str
        :param parameters: parameters of pipeline
        :type parameters: dict[str, str]
        :return: The submitted pipeline run.
        :rtype: azureml.pipeline.wrapper.pipeline_run.PipelineRun
        """
        workspace = self._workspace
        default_parameters = False
        if parameters is None:
            parameters = self._parameters
            default_parameters = True
            print('Submit pipeline {0} use default parameters {1}'.
                  format(self.name, self._parameters))
        telemetry_values = _get_telemetry_value_from_workspace(workspace)
        telemetry_values.update({
            'pipeline_id': self._id,
            'experiment_name': experiment_name,
            'default_parameters': default_parameters,
        })
        telemetry_values.update(_get_telemetry_value_from_pipeline_parameter(parameters))
        _LoggerFactory.trace(_get_logger(), "PublishedPipeline_submit", telemetry_values)
        from ._backend_provider import PublishedPipelineProviderContext
        context = PublishedPipelineProviderContext(
            pipeline_id=self._id, experiment_name=experiment_name,
            pipeline_description=description, pipeline_parameters=parameters)
        return self._backend_provider. \
            submit_published_pipeline(workspace=workspace, context=context)
