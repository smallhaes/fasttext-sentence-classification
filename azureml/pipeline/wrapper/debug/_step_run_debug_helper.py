# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import json
import re
import urllib.request
from pathlib import Path

from azureml.core import Experiment, Run, Dataset
from azureml.core.workspace import Workspace
from azureml.core.datastore import Datastore

from azureml.pipeline.wrapper._module_run_helper import _download_snapshot
from azureml.pipeline.wrapper._restclients.service_caller import DesignerServiceCaller
from azureml.pipeline.wrapper._utils import _sanitize_python_variable_name
from azureml.pipeline.wrapper.debug._constants import VSCODE_DIR, INPUT_DIR, LAUNCH_CONFIG, CONTAINER_DIR, \
    CONTAINER_CONFIG, SUBSCRIPTION_KEY, RESOURCE_GROUP_KEY, WORKSPACE_KEY, \
    LOCAL_MOUNT_DIR, DATA_REF_PREFIX, REMOTE_MOUNT_DIR, EXPERIMENT_KEY, RUN_KEY, ID_KEY
from azureml.pipeline.wrapper.debug._image import ImageBase
from azureml.pipeline.wrapper.dsl._utils import logger, _print_step_info


class DebugStepRunHelper:
    @staticmethod
    def installed_requirements():
        exc_map = {
            'docker': "https://www.docker.com/",
            'code': "https://code.visualstudio.com/Download"
        }
        _print_step_info(["Required {} can be installed here: {}".format(exc, url) for exc, url in exc_map.items()])

    @staticmethod
    def create_launch_config(step_name, python_path, commands, arguments):
        default_config = {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
        }
        if '-m' in commands:
            default_config['module'] = commands[-1]
        else:
            default_config['program'] = commands[-1]
        default_config['args'] = arguments
        default_config['pythonPath'] = python_path
        # create python debug config
        with open(INPUT_DIR / LAUNCH_CONFIG) as container_config:
            data = json.load(container_config)
            data['configurations'].append(default_config)
            os.makedirs(VSCODE_DIR, exist_ok=True)
            launch_config_path = os.path.join(VSCODE_DIR, LAUNCH_CONFIG)
            with open(launch_config_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)
        _print_step_info(f'Created launch config {launch_config_path} for step {step_name}')
        return launch_config_path

    @staticmethod
    def create_container_config(**kwargs):
        # create container config
        with open(INPUT_DIR / CONTAINER_CONFIG) as container_config:
            data = json.load(container_config)
            for key, val in kwargs.items():
                data[key] = val
            os.makedirs(CONTAINER_DIR, exist_ok=True)
            container_config_path = os.path.join(CONTAINER_DIR, CONTAINER_CONFIG)
            with open(container_config_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)
        return container_config_path


class DebugOnlineStepRunHelper(DebugStepRunHelper):
    @staticmethod
    def parse_designer_url(url):
        args = {}
        entries = re.split(r'[/&?]', url)
        try:
            for i, entry in enumerate(entries):
                if entry == EXPERIMENT_KEY:
                    if entries[i + 1] == ID_KEY:
                        args[EXPERIMENT_KEY] = entries[i + 2]
                    else:
                        args[EXPERIMENT_KEY] = entries[i + 1]
                elif entry in [RUN_KEY, WORKSPACE_KEY, RESOURCE_GROUP_KEY, SUBSCRIPTION_KEY]:
                    args[entry] = entries[i + 1]
        except BaseException as e:
            raise RuntimeError(f'Failed to parse portal url: {url}') from e

        return args.get(RUN_KEY, None), args.get(EXPERIMENT_KEY, None), args.get(WORKSPACE_KEY, None), args.get(
            RESOURCE_GROUP_KEY, None), args.get(SUBSCRIPTION_KEY, None)

    @staticmethod
    def get_pipeline_run(run_id, experiment_name, workspace_name, resource_group_name, subscription_id):
        workspace = Workspace(subscription_id, resource_group_name, workspace_name)

        experiments = [experiment for experiment in Experiment.list(workspace) if
                       experiment.id == experiment_name or experiment.name == experiment_name]
        assert len(experiments) > 0, "Experiment %s not found" % experiment_name
        experiment = experiments[0]

        pipeline_run = Run(experiment, run_id)

        _print_step_info(
            f'Workspace: {workspace.name} Experiment: {experiment_name} StepRun: {run_id}')

        return pipeline_run

    @staticmethod
    def get_image_id(step_name, details):
        if 'properties' not in details:
            raise RuntimeError(f'{step_name} does not have properties')
        properties = details['properties']
        if 'AzureML.DerivedImageName' in properties:
            return properties['AzureML.DerivedImageName']
        else:
            for log_file, url in details['logFiles'].items():
                if 'azureml-execution' in log_file:
                    content = urllib.request.urlopen(url).read()
                    m = re.findall(r'latest: Pulling from (azureml/azureml_[^\\n]+)', str(content))
                    if len(m) > 0:
                        return m[0]
                    m = re.findall(r'Start to pulling docker image: [^/]+/(azureml/azureml_[^\\n]+)', str(content))
                    if len(m) > 0:
                        return m[0]
            raise RuntimeError(f'{step_name} does not have valid logs with image pattern azureml/azureml_[^\\n]')

    @staticmethod
    def prepare_dev_container(workspace, step, dry_run=False) -> ImageBase:
        # prepare image
        try:
            environment = step.get_environment()
        except Exception as e:
            original_error_message = f'{e.__class__.__name__}: {e}'
            raise RuntimeError('Failed to get environment details from step run details, '
                               'please make sure this step run has started successfully.\n'
                               f'Original error: {original_error_message}') from e
        image_details = environment.get_image_details(workspace)
        step_run_image = ImageBase(image_details)
        if not dry_run:
            step_run_image.pull_module_image()

        # create container config
        data = {'image': step_run_image.image_name, 'appPort': 8090}
        DebugOnlineStepRunHelper.create_container_config(**data)
        return step_run_image

    @staticmethod
    def download_snapshot(service_caller: DesignerServiceCaller, run_id: str, step_run_id: str, dry_run=False):
        if dry_run:
            return
        # TODO: move service caller to debugger and use run detail as param
        run_details = service_caller.get_pipeline_run_step_details(run_id, step_run_id, include_snaptshot=True)
        snapshot_url = run_details.snapshot_info.root_download_url
        snapshot_path = os.getcwd()
        _download_snapshot(snapshot_url, snapshot_path)
        _print_step_info(f'Downloaded snapshot {snapshot_path}')

    @staticmethod
    def prepare_inputs(workspace, details, dry_run=False):
        if 'runDefinition' not in details:
            raise RuntimeError('Failed to get runDefinition from step run details, '
                               'please make sure this step run has started successfully.')
        port_arg_map = {}
        partial_success = False
        # data reference
        data_references = details['runDefinition']['dataReferences']
        for data_reference_name, data_store in data_references.items():
            data_store_name = data_store['dataStoreName']
            path_on_data_store = data_store['pathOnDataStore']
            path_on_data_store = _sanitize_python_variable_name(path_on_data_store)
            port_arg_map[data_reference_name] = path_on_data_store
            if not dry_run:
                result = DebugOnlineStepRunHelper.download_data_reference(workspace, data_store_name,
                                                                          path_on_data_store)
                if not result:
                    partial_success = True

        # dataset
        dataset = details['runDefinition']['data']
        for dataset_name, data in dataset.items():
            dataset_id = data['dataLocation']['dataset']['id']
            if not dry_run:
                result = DebugOnlineStepRunHelper.download_dataset(workspace, dataset_id)
                if not result:
                    partial_success = True
            port_arg_map[dataset_name] = dataset_id
        _print_step_info(f'Downloaded data: {port_arg_map}')

        return port_arg_map, partial_success

    @staticmethod
    def download_data_reference(workspace, data_store_name, path_on_data_store) -> bool:
        try:
            blob_data_store = Datastore.get(workspace, data_store_name)
            blob_data_store.download(target_path=str(LOCAL_MOUNT_DIR), prefix=path_on_data_store, overwrite=False)
            # output directory might be empty
            if not Path(LOCAL_MOUNT_DIR / path_on_data_store).exists():
                os.makedirs(LOCAL_MOUNT_DIR / path_on_data_store)
            return True
        except Exception as e:
            logger.warning('Could not download dataset {} due to error {}'.format(path_on_data_store, e))
            return False

    @staticmethod
    def download_dataset(workspace, dataset_id) -> bool:
        try:
            dataset = Dataset.get_by_id(workspace, dataset_id)
            target_path = str(LOCAL_MOUNT_DIR / dataset_id)
            dataset.download(target_path=target_path, overwrite=True)
            return True
        except Exception as e:
            logger.warning('Could not download dataset {} due to error {}'.format(dataset_id, e))
            return False

    @staticmethod
    def prepare_arguments(step_name, details, port_arg_map):
        if 'runDefinition' not in details:
            raise RuntimeError('Failed to get runDefinition from step run details, '
                               'please make sure this step run has started successfully.')
        run_definition = details['runDefinition']
        arguments = run_definition['arguments']
        environment_vars = run_definition['environment']['environmentVariables']
        environment_vars = {f'${key}': environment_vars[key] for key in environment_vars}
        for data_reference_name, port_dir in port_arg_map.items():
            data_reference_constant = DATA_REF_PREFIX + data_reference_name
            data_reference_path = str(REMOTE_MOUNT_DIR / port_dir)
            environment_vars[data_reference_constant] = data_reference_path

        arguments = [x.replace(x, environment_vars[x]) if x in environment_vars else x
                     for x in arguments]
        _print_step_info(f'Prepared arguments: {arguments} for step {step_name}')
        run_definition = details['runDefinition']
        script = run_definition['script']
        return script, arguments
