# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines classes for submitted pipeline runs, including classes for checking status and retrieving run details."""

import json
import os
import re
import sys
import time
import yaml

from azureml._base_sdk_common._docstring_wrapper import experimental
from azureml.core.datastore import Datastore
from azureml.core import Run, ScriptRun
from azureml.data.datapath import DataPath
from azureml.exceptions import ExperimentExecutionException, ActivityFailedException
from azureml._execution import _commands
from azureml._restclient.utils import create_session_with_retry

from ._loggerfactory import _LoggerFactory, _PUBLIC_API, track
from ._restclients.designer.designer_service_client import DesignerServiceClient
from ._restclients.designer.models.designer_service_client_enums import RunStatus
from ._restclients.service_caller import DesignerServiceCaller
from ._telemetry import _get_telemetry_value_from_workspace
from ._utils import _is_prod_workspace, _in_jupyter_nb

RUNNING_STATES = [RunStatus.not_started, RunStatus.starting, RunStatus.provisioning,
                  RunStatus.preparing, RunStatus.queued, RunStatus.running]
# align with UX query status interval
REQUEST_INTERVAL_SECONDS = 5

_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


class PipelineRun(Run):
    """Represents a run of a :class:`azureml.pipeline.wrapper.Pipeline`.

    :param experiment: The experiment object associated with the pipeline run.
    :type experiment: azureml.core.Experiment
    :param run_id: The run ID of the pipeline run.
    :type run_id: str
    """

    def __init__(self, experiment, run_id):
        """Initialize a Pipeline run.

        :param experiment: The experiment object associated with the pipeline run.
        :type experiment: azureml.core.Experiment
        :param run_id: The run ID of the pipeline run.
        :type run_id: str
        """
        super(PipelineRun, self).__init__(experiment, run_id)
        from ._backend_provider import SmtBackendProvider
        self._backend_provider = SmtBackendProvider()

    def _get_all_runs_status_and_logs(self, child_runs: list = None):
        run_graph_status = self._backend_provider.get_pipeline_run_status(
            self.experiment, self._run_id)
        parent_status = run_graph_status.status
        graph_node_status = run_graph_status.graph_nodes_status
        children = self.get_children(_rehydrate_runs=False) if child_runs is None else child_runs
        logs = {}
        urls = {}
        processed_step_runs = []
        for step_run in children:
            if step_run.id not in processed_step_runs:
                processed_step_runs.append(step_run.id)
                _step_run = StepRun(self.experiment, step_run_id=step_run.id)
                logs[_step_run.tags['azureml.nodeid']] = _step_run.get_details()['logFiles']
                urls[_step_run.tags['azureml.nodeid']] = step_run._run_details_url

        def _graph_node_status_to_json(node_id, status, url_dict):
            return {'status': status.status,
                    'statusCode': status.status_code,
                    'startTime': None if status.start_time is None else status.start_time.isoformat(),
                    'endTime': None if status.end_time is None else status.end_time.isoformat(),
                    'runStatus': status.run_status,
                    'runDetailsUrl': url_dict.get(node_id),
                    'statusDetail': status.status_detail}

        node_status = {k: _graph_node_status_to_json(k, v, urls) for k, v in graph_node_status.items()}
        node_logs = {k: v for k, v in logs.items()}

        # append parent run status and logs
        node_status.update({'@parent': {'runStatus': parent_status.run_status,
                                        'runDetailsUrl': self.get_portal_url(),
                                        'statusDetail': parent_status.status_detail,
                                        'startTime': parent_status.start_time,
                                        'endTime': parent_status.end_time}})
        node_logs.update({'@parent': self.get_details()['logFiles']})

        return node_status, node_logs

    @property
    def workspace(self):
        """Return the workspace containing the experiment.

        :return: The workspace object.
        :rtype: azureml.core.workspace.Workspace
        """
        return self.experiment.workspace

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name="PipelineRun_publish")
    def publish(self, name: str, description: str = None, tags: dict = None):
        """
        Publish a pipeline run and make it available for rerunning.

        You can get the pipeline rest endpoint from the PublishedPipeline object returned by this function. With the
        rest endpoint, you can invoke the pipeline from external applications using REST calls. For information
        about how to authenticate when calling REST endpoints, see https://aka.ms/pl-restep-auth.

        The original pipeline associated with the pipeline run is used as the base for the published pipeline.

        :param name: The name of the published pipeline.
        :type name: str
        :param description: The description of the published pipeline.
        :type description: str
        :param tags: tags of pipeline to publish
        :type tags: dict[str, str]
        :return: Created published pipeline.
        :rtype: azureml.pipeline.wrapper._published_pipeline.PublishedPipeline
        """
        experiment_name = self.experiment.name
        service_caller = DesignerServiceCaller(self.workspace)

        from ._backend_provider import PublishedPipelineProviderContext
        context = PublishedPipelineProviderContext(
            run_id=self.id, pipeline_name=name,
            pipeline_description=description,
            tags=tags, use_pipeline_endpoint=False, properties=self.properties,
            experiment_name=experiment_name,
            graph=service_caller.get_pipeline_run_graph(
                experiment_name=experiment_name, pipeline_run_id=self.id).graph)
        result = self._backend_provider. \
            publish_pipeline(workspace=self.workspace,
                             context=context)
        telemetry_values = _get_telemetry_value_from_workspace(self.workspace)
        telemetry_values.update({
            'run_id': self.id,
            'pipeline_id': result.id,
            'use_pipeline_endpoint': False,
        })
        _LoggerFactory.trace(_get_logger(), "PipelineRun_publish", telemetry_values)
        return result

    @experimental
    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='PipelineRun_publish')
    def publish_to_endpoint(self, name: str, pipeline_endpoint_name: str,
                            description: str = None, pipeline_endpoint_description: str = None,
                            set_as_default: bool = True, use_existing_pipeline_endpoint: bool = True,
                            tags: dict = None):
        """
        Publish a pipeline run to pipeline_endpoint.

        A pipeline enpoint is a :class:`azureml.pipeline.wrapper.Pipeline` workflow
         that can be triggered from a unique endpoint URL.

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

        :return: Created published pipeline inside pipeline endpoint.
        :rtype: azureml.pipeline.wrapper._published_pipeline.PublishedPipeline
        """
        experiment_name = self.experiment.name
        service_caller = DesignerServiceCaller(self.workspace)

        from ._backend_provider import PublishedPipelineProviderContext
        context = PublishedPipelineProviderContext(
            run_id=self.id, experiment_name=experiment_name,
            pipeline_name=name, pipeline_description=description,
            pipeline_endpoint_name=pipeline_endpoint_name,
            pipeline_endpoint_description=pipeline_endpoint_description,
            tags=tags, set_as_default=set_as_default,
            use_existing_pipeline_endpoint=use_existing_pipeline_endpoint,
            use_pipeline_endpoint=True, properties=self.properties,
            graph=service_caller.get_pipeline_run_graph(
                experiment_name=experiment_name, pipeline_run_id=self.id).graph)
        result = self._backend_provider. \
            publish_pipeline(workspace=self.workspace, context=context)
        telemetry_values = _get_telemetry_value_from_workspace(self.workspace)
        telemetry_values.update({
            'run_id': self.id,
            'pipeline_id': result.id,
            'use_pipeline_endpoint': True,
            'set_as_default': set_as_default,
            'use_existing_pipeline_endpoint': use_existing_pipeline_endpoint,
        })
        _LoggerFactory.trace(_get_logger(), "PipelineRun_publish", telemetry_values)
        return result

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='PipelineRun_waitForCompletion')
    def wait_for_completion(self, show_output=False, show_graph=True,
                            timeout_seconds=sys.maxsize, raise_on_error=True):
        """
        Wait for the completion of this pipeline run.

        Returns the status after the wait.

        :param show_output: Indicates whether to show the pipeline run status on sys.stdout.
        :type show_output: bool
        :param show_graph: Indicates whether to show the graph with run status on notebook.
         If not in notebook environment, overwrite this value to False
        :type show_graph: bool
        :param timeout_seconds: The number of seconds to wait before timing out.
        :type timeout_seconds: int
        :param raise_on_error: Indicates whether to raise an error when the run is in a failed state.
        :type raise_on_error: bool

        :return: The final status.
        :rtype: str
        """
        print('PipelineRunId:', self.id)
        print('Link to Azure Machine Learning Portal:', self.get_portal_url())

        start_time = time.time()
        status = self._get_run_status()

        if not _in_jupyter_nb() and show_graph:
            print('Could not show graph with notebook environment. Fall back to show output on console.')
            show_graph = False
            show_output = True

        if show_graph:
            from ._widgets._visualize import _visualize
            graph_json = self.visualization_builder.build_visualization_dict()
            is_prod = _is_prod_workspace(self.workspace)
            visualizer = _visualize(graph_json, is_prod=is_prod)

        def update_graph_status(visualizer, child_runs: list = None):
            node_status, node_logs = self._get_all_runs_status_and_logs(child_runs)
            visualizer.send_message(message='status', content=node_status)
            visualizer.send_message(message='logs', content=node_logs)

        def timeout(start_time: float, timeout_seconds: float):
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                print('Timed out of waiting. Elapsed time:', elapsed_time)
                return True
            return False

        while status in RUNNING_STATES:
            children = self.get_children(_rehydrate_runs=False)
            if show_output:
                try:
                    old_status = None
                    processed_step_runs = []
                    if not status == old_status:
                        old_status = status
                        print('PipelineRun Status:', status.value)
                    for step_run in children:
                        if step_run.id not in processed_step_runs:
                            processed_step_runs.append(step_run.id)
                            time_elapsed = time.time() - start_time
                            print('\n')
                            _step_run = StepRun(self.experiment, step_run_id=step_run.id)
                            _step_run.wait_for_completion(timeout_seconds=timeout_seconds - time_elapsed,
                                                          raise_on_error=raise_on_error)
                            print('')
                except KeyboardInterrupt:
                    error_message = "The output streaming for the run interrupted.\n" \
                                    "But the run is still executing on the compute target. \n" \
                                    "Details for canceling the run can be found here: " \
                                    "https://aka.ms/aml-docs-cancel-run"
                    raise ExperimentExecutionException(error_message)
            if show_graph:
                update_graph_status(visualizer, children)

            if timeout(start_time, timeout_seconds):
                break
            time.sleep(REQUEST_INTERVAL_SECONDS)
            status = self._get_run_status()

        final_details = self.get_details()
        warnings = final_details.get("warnings")
        error = final_details.get("error")

        if show_output:
            summary_title = '\nPipelineRun Execution Summary'
            print(summary_title)
            print('=' * len(summary_title))
            print('PipelineRun Status:', status.value)
            if warnings:
                messages = [x.get("message") for x in warnings if x.get("message")]
                if len(messages) > 0:
                    print('\nWarnings:')
                    for message in messages:
                        print(message)
            if error:
                print('\nError:')
                print(json.dumps(error, indent=4))

            print(final_details)
            print('', flush=True)

        if show_graph:
            # do one more extra update to ensure the final status is sent
            update_graph_status(visualizer)

        # put the raise logic after show_graph to ensure the last update
        if error and raise_on_error:
            raise ActivityFailedException(error_details=json.dumps(error, indent=4))

        return status

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='PipelineRun_getStatus')
    def get_status(self):
        """
        Fetch the latest status of the run.

        :return: The latest status.
        :rtype: str
        """
        return self._get_run_status().value

    def _get_run_status(self):
        run_status_entity = self._backend_provider.get_pipeline_run_status(self.experiment, self._run_id)
        return list(RunStatus)[int(run_status_entity.status.run_status)]

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='PipelineRun_findStepRun')
    def find_step_run(self, name):
        """Find a step run in the pipeline by name.

        :param name: The name of the step to find.
        :type name: str

        :return: List of :class:`azureml.pipeline.wrapper.StepRun` objects with the provided name.
        :rtype: builtin.list
        """
        children = self.get_children()
        step_runs = []
        for child in children:
            if name == child._run_dto['name']:
                step_run = StepRun(child.experiment, child.id)
                step_runs.append(step_run)

        return step_runs


class InputPort(object):
    """Represents a input port of a :class:`azureml.pipeline.wrapper.Module`.

    This class can be used to manage the output of the port.

    :param name: The name of the port.
    :type name: str.
    :param _type: The type of the port.
    :type _type: str.
    :param step_run: The associated StepRun object.
    :type step_run: azureml.pipeline.wrapper.StepRun
    """

    def __init__(self, name, _type, step_run):
        """Initialize the Port object.

        :param name: The name of the port.
        :type name: str.
        :param _type: The type of the port.
        :type _type: builtin.list.
        :param step_run: The associated StepRun object.
        :type step_run: azureml.pipeline.wrapper.StepRun

        """
        self._name = name
        self._type = _type
        self._step_run = step_run
        self._workspace = self._step_run.experiment.workspace
        self._designerServiceClient = DesignerServiceClient(self._workspace.service_context._get_pipelines_url())

    def __str__(self):
        """Return the description of a Port object."""
        return "Port(Name:{},\nType:{},\nStepRun:{})".format(self._name, self._type, self._step_run)

    def __repr__(self):
        """Return str(self)."""
        return self.__str__()

    @property
    def name(self):
        """Return the port name.

        The name of a port.

        :return: The port name.
        :rtype: str
        """
        return self._name

    @property
    def type(self):
        """Return the port type.

        The type of a port.

        :return: The port type.
        :rtype: str
        """
        return self._type

    @property
    def step_run(self):
        """Return the StepRun object.

        A StepRun object is associated with a Port object.

        :return: The StepRun object.
        :rtype: azureml.pipeline.wrapper.StepRun
        """
        return self._step_run

    @property
    def _step_run_outputs(self):
        """Return the PipelineStepRunOutputs object.

        The PipelineStepRunOutputs object contains much information about the pipeline.

        :return: The PipelineStepRunOutputs object.
        :rtype: azureml.pipeline.wrapper._restclients.designer.models
        .pipeline_step_run_outputs_py3.PipelineStepRunOutputs
        """
        designerServiceCaller = DesignerServiceCaller(self._workspace)
        return self._designerServiceClient.pipeline_runs.get_pipeline_run_step_outputs(
            self._workspace.subscription_id, self._workspace.resource_group, self._workspace.name,
            self._step_run.properties['azureml.pipelinerunid'], self._step_run.tags['azureml.nodeid'],
            self._step_run.id, designerServiceCaller._get_custom_headers())

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Port_isType')
    def is_type(self, port_type):
        """Check whether port_type belongs to the port type.

        :param port_type: A string indicating the port type.
        :type port_type: str
        :return: The boolean value of whether _type belongs to the port type.
        :rtype: bool
        """
        return port_type in self._type

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Port_download')
    def download(self, local_path, overwrite=False, show_progress=True):
        """Download the data associated with the Port object."""
        raise NotImplementedError("Unable to download data from input port.")

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Port_getDataPath')
    def get_data_path(self):
        """Get the DataPath object associated with the specific port."""
        raise NotImplementedError("Unable to get data path for input port.")


class OutputPort(object):
    """Represents a output port of a :class:`azureml.pipeline.wrapper.Module`.

    This class can be used to manage the output of the port.

    :param name: The name of the port.
    :type name: str.
    :param _type: The type of the port.
    :type _type: str.
    :param step_run: The associated StepRun object.
    :type step_run: azureml.pipeline.wrapper.StepRun
    """

    def __init__(self, name, _type, step_run):
        """Initialize the Port object.

        :param name: The name of the port.
        :type name: str.
        :param _type: The type of the port.
        :type _type: builtin.list.
        :param step_run: The associated StepRun object.
        :type step_run: azureml.pipeline.wrapper.StepRun

        """
        self._name = name
        self._type = _type
        self._step_run = step_run
        self._workspace = self._step_run.experiment.workspace
        self._designerServiceClient = DesignerServiceClient(self._workspace.service_context._get_pipelines_url())

    def __str__(self):
        """Return the description of a Port object."""
        return "Port(Name:{},\nType:{},\nStepRun:{})".format(self._name, self._type, self._step_run)

    def __repr__(self):
        """Return str(self)."""
        return self.__str__()

    @property
    def name(self):
        """Return the port name.

        The name of a port.

        :return: The port name.
        :rtype: str
        """
        return self._name

    @property
    def type(self):
        """Return the port type.

        The type of a port.

        :return: The port type.
        :rtype: str
        """
        return self._type

    @property
    def step_run(self):
        """Return the StepRun object.

        A StepRun object is associated with a Port object.

        :return: The StepRun object.
        :rtype: azureml.pipeline.wrapper.StepRun
        """
        return self._step_run

    @property
    def _step_run_outputs(self):
        """Return the PipelineStepRunOutputs object.

        The PipelineStepRunOutputs object contains much information about the pipeline.

        :return: The PipelineStepRunOutputs object.
        :rtype: azureml.pipeline.wrapper._restclients.designer.models
        .pipeline_step_run_outputs_py3.PipelineStepRunOutputs
        """
        designerServiceCaller = DesignerServiceCaller(self._workspace)
        return self._designerServiceClient.pipeline_runs.get_pipeline_run_step_outputs(
            self._workspace.subscription_id, self._workspace.resource_group, self._workspace.name,
            self._step_run.properties['azureml.pipelinerunid'], self._step_run.tags['azureml.nodeid'],
            self._step_run.id, designerServiceCaller._get_custom_headers())

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Port_isType')
    def is_type(self, port_type):
        """Check whether port_type belongs to the port type.

        :param port_type: A string indicating the port type.
        :type port_type: str
        :return: The boolean value of whether _type belongs to the port type.
        :rtype: bool
        """
        return port_type in self._type

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Port_download')
    def download(self, local_path, overwrite=False, show_progress=True):
        """Download the data associated with the Port object.

        :param local_path: Local path to download to.
        :type local_path: str
        :param overwrite: Indicates whether to overwrite existing files. Defaults to False.
        :type overwrite: bool, optional
        :param show_progress: Indicates whether to show the progress of the download in the console.
            Defaults to True.
        :type show_progress: bool, optional
        :return: The path where the files are saved.
        :rtype: str
        """
        data_path = self.get_data_path()
        if not data_path:
            return 'No outputs in port "{}"'.format(self._name)
        data_path._datastore.download(target_path=local_path, prefix=data_path.path_on_datastore,
                                      overwrite=overwrite,
                                      show_progress=show_progress)
        saved_path = os.path.join(local_path, data_path.path_on_datastore)
        return saved_path

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='Port_getDataPath')
    def get_data_path(self):
        """Get the DataPath object associated with the specific port.

        :return: The DataPath object associated with the specific port.
        :rtype: azureml.data.datapath.DataPath
        """
        module_name_for_dir = re.sub(pattern=' ', repl='_', string=self._name)
        port_outputs = self._step_run_outputs.port_outputs
        if module_name_for_dir not in port_outputs:
            return None
        port_output = port_outputs[module_name_for_dir]
        path_on_datastore = port_output.relative_path
        data_store_name = port_output.data_store_name
        datastore = Datastore.get(self._workspace, data_store_name)
        data_path = DataPath(datastore=datastore, path_on_datastore=path_on_datastore)
        return data_path


class StepRun(Run):
    """Represents a run of a step in a :class:`azureml.pipeline.wrapper.Pipeline`.

    This class can be used to obtain run details once the parent pipeline run is submitted
    and the pipeline has submitted the step run.

    :param experiment: The experiment object of the step run.
    :type experiment: azureml.core.Experiment
    :param step_run_id: The run ID of the step run.
    :type step_run_id: str
    """

    def __init__(self, experiment, step_run_id):
        """Initialize a StepRun.

        :param experiment: The experiment object of the step run.
        :type experiment: azureml.core.Experiment
        :param step_run_id: The run ID of the step run.
        :type step_run_id: str
        """
        super(self.__class__, self).__init__(experiment, step_run_id)
        self._workspace = experiment.workspace
        self._input_ports = None
        self._output_ports = None

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='StepRun_waitForCompletion')
    def wait_for_completion(self, show_output=True, timeout_seconds=sys.maxsize, raise_on_error=True):
        """Wait for the completion of this step run.

        Returns the status after the wait.

        :param show_output: show_output=True shows the pipeline run status on sys.stdout.
        :type show_output: bool
        :param timeout_seconds: Number of seconds to wait before timing out.
        :type timeout_seconds: int
        :param raise_on_error: Indicates whether to raise an error when the Run is in a failed state.
        :type raise_on_error: bool

        :return: The final status.
        :rtype: str
        """
        print('StepRunId:', self.id)
        print('Link to Azure Machine Learning Portal:', self.get_portal_url())

        if show_output:
            try:
                return self._stream_run_output(timeout_seconds=timeout_seconds,
                                               raise_on_error=raise_on_error)
            except KeyboardInterrupt:
                error_message = "The output streaming for the run interrupted.\n" \
                                "But the run is still executing on the compute target. \n" \
                                "Details for canceling the run can be found here: " \
                                "https://aka.ms/aml-docs-cancel-run"

                raise ExperimentExecutionException(error_message)
        else:
            status = self.get_status()
            while status in RUNNING_STATES:
                time.sleep(REQUEST_INTERVAL_SECONDS)
                status = self.get_status()

            final_details = self.get_details()
            error = final_details.get("error")
            if error and raise_on_error:
                raise ActivityFailedException(error_details=json.dumps(error, indent=4))

            return status

    def _stream_run_output(self, timeout_seconds=sys.maxsize, raise_on_error=True):
        """Stream the experiment run output to the specified file handle.

        :param timeout_seconds: Number of seconds to wait before timing out.
        :type sys.timeout_seconds: int
        :param raise_on_error: Indicates whether to raise an error when the Run is in a failed state
        :type raise_on_error: bool
        :return: The status of the run
        :rtype: str
        """
        def incremental_print(log, num_printed):
            count = 0
            for line in log.splitlines():
                if count >= num_printed:
                    print(line)
                    num_printed += 1
                count += 1
            return num_printed

        num_printed_lines = 0
        current_log = None

        start_time = time.time()
        session = create_session_with_retry()

        old_status = None
        status = self.get_status()
        while status in RUNNING_STATES:
            if not status == old_status:
                old_status = status
                print('StepRun(', self.name, ') Status:', status)

            current_details = self.get_details()
            available_logs = [x for x in current_details["logFiles"]
                              if re.match(r"azureml-logs/[\d]{2}.+\.txt", x)]
            available_logs.sort()
            next_log = ScriptRun._get_last_log_primary_instance(available_logs) if available_logs else None

            if available_logs and current_log != next_log:
                num_printed_lines = 0
                current_log = next_log
                print("\nStreaming " + current_log)
                print('=' * (len(current_log) + 10))

            if current_log:
                current_log_uri = current_details["logFiles"].get(current_log)
                if current_log_uri:
                    content = _commands._get_content_from_uri(current_log_uri, session)
                    num_printed_lines = incremental_print(content, num_printed_lines)

            if time.time() - start_time > timeout_seconds:
                print('Timed out of waiting. Elapsed time:', time.time() - start_time)
                break
            time.sleep(REQUEST_INTERVAL_SECONDS)
            status = self.get_status()

        summary_title = '\nStepRun(' + self.name + ') Execution Summary'
        print(summary_title)
        print('=' * len(summary_title))
        print('StepRun(', self.name, ') Status:', status)

        final_details = self.get_details()
        warnings = final_details.get("warnings")
        if warnings:
            messages = [x.get("message") for x in warnings if x.get("message")]
            if len(messages) > 0:
                print('\nWarnings:')
                for message in messages:
                    print(message)

        error = final_details.get("error")
        if error and not raise_on_error:
            print('\nError:')
            print(json.dumps(error, indent=4))
        if error and raise_on_error:
            raise ActivityFailedException(error_details=json.dumps(error, indent=4))

        print(final_details)
        print('', flush=True)

        return status

    def _get_yamlModuleDefinition(self):
        """Get the _YamlModuleDefinition object.

        :return: The _YamlModuleDefinition object.
        :rtype: azureml.pipeline.wrapper.dsl._module_spec._YamlModuleDefinition
        """
        from azureml.pipeline.wrapper.dsl._module_spec import _YamlModuleDefinition
        designerServiceCaller = DesignerServiceCaller(self._workspace)
        yaml_str = designerServiceCaller.get_module_yaml_by_id(module_id=self.properties['azureml.moduleid'])
        yaml_dict = yaml.load(yaml_str, Loader=yaml.FullLoader)
        yamlModuleDefinition = _YamlModuleDefinition(yaml_dict)
        return yamlModuleDefinition

    def _create_ports(self):
        """Get the list of input ports and output ports separately."""
        yamlModuleDefinition = self._get_yamlModuleDefinition()
        input_ports = yamlModuleDefinition.input_ports
        if input_ports:
            self._input_ports = []
            for port in input_ports:
                port_type = [port['type']] if type(port['type']) == str else port['type']
                self._input_ports.append(InputPort(name=port['name'], _type=port_type, step_run=self))
        output_ports = yamlModuleDefinition.output_ports
        if output_ports:
            self._output_ports = []
            for port in output_ports:
                port_type = [port['type']] if type(port['type']) == str else port['type']
                self._output_ports.append(OutputPort(name=port['name'], _type=port_type, step_run=self))

    @property
    def input_ports(self):
        """Get a list of input ports."""
        if not self._input_ports and not self._output_ports:
            self._create_ports()
        return self._input_ports

    @property
    def output_ports(self):
        """Get a list of output ports."""
        if not self._input_ports and not self._output_ports:
            self._create_ports()
        return self._output_ports

    @track(_get_logger, activity_type=_PUBLIC_API, activity_name='StepRun_getPortByName')
    def get_port(self, name):
        """Get a Port object by name.

        :param name: The name of the port.
        :type name: str
        :return: The Port object.
        :rtype: azureml.pipeline.wrapper.Port
        """
        for port in self.output_ports + self.input_ports:
            if name == port.name or ' '.join(name.split('_')) == port.name:
                return port
        return None
