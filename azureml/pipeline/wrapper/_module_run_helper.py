# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os
import sys
import json
import requests
import zipfile
import subprocess
import tarfile
import copy
import traceback
from threading import Lock, currentThread, main_thread
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from io import BytesIO

from azureml.core.environment import Environment
from azureml.pipeline.wrapper._restclients.service_caller import DesignerServiceCaller
from azureml._model_management._util import write_dir_in_container
from azureml._model_management._util import write_file_in_container
from azureml._model_management._util import get_docker_client
from azureml.core import Datastore, Dataset
from azureml.data.datapath import DataPath
from azureml.data.data_reference import DataReference
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data import FileDataset
from ._pipeline_parameters import PipelineParameter


PREFIX_PARAMETER = 'AZUREML_PARAMETER_'
PREFIX_DATAREFERENCE = 'AZUREML_DATAREFERENCE_'
CONTAINER_INPUT_PATH = '/mnt/input'
CONTAINER_OUTPUT_PATH = '/mnt/output'
CONTAINER_MOUNT_SCRIPTS_PATH = '/mnt/scripts'
SCRIPTE_DIR_NAME = 'scripts'
OUTPUT_DIR_NAME = 'outputs'
EXECUTION_LOGFILE = 'excutionlogs.txt'
MODULE_LOGFILE = "module-log.txt"
RUN_STATUS = {
    'NotStarted': 0,
    'Preparing': 3,
    'Running': 5,
    'Completed': 8,
    'Failed': 9,
    'Cancel': 10
}


def _module_run(module, working_dir, use_docker, run=None, node_id=None, visualizer=None, show_output=True,
                module_to_node_mapping={}, data_dir=None, pipeline_parameters=None):
    """
    Run module

    _module_run will run module in local environment/container. In prepare state, will download module
    snapshots and input dataste, generate module execute command and pull module image. Then will execute
    command in local environment/container.

    :param module: Executed module
    :type module: azureml.pipeline.wrapper.Module
    :param working_dir: module data and snapshot store path
    :type working_dir: str
    ::param use_docker: If use_docker=True, will pull image from azure and run module in container.
                        If use_docker=False, will directly run module script.
    :type use_docker: bool
    :param run: pipeline run used for tracking run history.
    :type run: azureml.core.run.Run
    :param node_id: Node id of module
    :type node_id: str
    :param visualizer: To show pipeline graph in notebook
    :type visualizer: azureml.pipeline.wrapper._widgets._visualize
    :param show_output: Indicates whether to show the pipeline run status on sys.stdout.
    :type show_output: bool
    :param module_to_node_mapping: Mapping of module to node info
    :type module_to_node_mapping: dict{(str, dict)}
    :param data_dir: If module input data in remote, will download to data_dir.
                     If data_dir=None, will set working_dir to data_dir.
    :type data_dir: str
    :param pipeline_parameters: An optional dictionary of pipeline parameter
    :type pipeline_parameters: dict{(str, object)}
    :return: Module run status
    :rtype: str
    """
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    if not data_dir:
        data_dir = working_dir
    is_main_thread = currentThread() is main_thread()
    logger = Logger(log_path=os.path.join(working_dir, EXECUTION_LOGFILE),
                    show_terminal=(show_output and is_main_thread))

    module_log_file_path = Path(working_dir) / MODULE_LOGFILE
    status = None
    log_file_url = None
    try:
        module = _prepare_module_run(module, data_dir, pipeline_parameters, module_to_node_mapping)
        # visualizer start
        module_run_status = 'Preparing'
        status = update_module_status(module_run_status, run_details_url=run._run_details_url if run else None)
        update_visualizer(visualizer, node_id, status)

        # Get snapshot directory of module
        print('Preparing snapshot')
        start_time = datetime.now()
        snapshot_path = os.path.join(working_dir, SCRIPTE_DIR_NAME)
        _get_module_snapshot(module, snapshot_path)
        print("Prepared snapshot, time elapsed {}".format(datetime.now() - start_time))

        if run:
            print('Preparing experiment and run in run history')
            start_time = datetime.now()
            run.take_snapshot(snapshot_path)
            print("Prepared experiment and run in run history, time elapsed {}".format(datetime.now() - start_time))
            print('RunId:', run.id)
            print('Link to Azure Machine Learning Portal:', run.get_portal_url())

        # Genterate command
        command, volumes, environment = _generate_command(module, working_dir, use_docker)
        # For metirc needed environment variables
        if run:
            environment.update(_set_environment_variables_for_run(run))

        if use_docker:
            # Add scripts in volumes
            volumes[snapshot_path] = {'bind': CONTAINER_MOUNT_SCRIPTS_PATH, 'mode': 'rw'}

            # Pull image to local
            from azureml.pipeline.wrapper.debug._image import ModuleImage
            module_image = ModuleImage(module)
            module_image.pull_module_image()
            image_name = module_image.image_name

            # Start executing module script
            print(f'Start running module {module.name}')
            module_run_status = 'Running'
            update_visualizer(visualizer, node_id, update_module_status(module_run_status, status), log_file_url)
            # Run container
            docker_client = get_docker_client()
            container, module_run_success, stdout = _run_docker_container(
                docker_client, image_name, command=command, volumes=volumes, environment=environment)
            with open(module_log_file_path, 'w') as file_obj:
                file_obj.write(stdout)
        else:
            print(f'Start running module {module.name}')
            module_run_status = 'Running'
            update_visualizer(visualizer, node_id, update_module_status(module_run_status, status), log_file_url)
            environment.update(os.environ)
            process = subprocess.run(
                command, env=environment, cwd=snapshot_path, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
            with open(module_log_file_path, 'w') as file_obj:
                for line in process.stdout:
                    file_obj.write(line)

            if process.returncode == 0:
                module_run_success = True
            else:
                module_run_success = False

        module_run_status = "Completed" if module_run_success else "Failed"
        print(f'Finish running module {module.name}, '
              f'module run status is {module_run_status}')

        # create module output file/folder which not exists
        check_module_run_output(module, volumes)

        # Upload module outputs and log file to portal
        if run:
            log_file_url = _update_run_log_and_output(
                run, module, working_dir, module_run_success, module_log_file_path)

        update_visualizer(visualizer, node_id, update_module_status(module_run_status, status), log_file_url)

        if use_docker and not module_run_success:
            _prepare_debug_config_for_module(
                module, module_image, command, volumes, environment, snapshot_path)

    except Exception as ex:
        traceback.print_exc()
        if run:
            run.fail(error_details=str(ex))
            print('Finish upload run status to runhistory')
        update_visualizer(
            visualizer,
            node_id,
            update_module_status('Failed', status, run_details_url=run._run_details_url if run else None),
            log_file_url)
        raise ex
    finally:
        execution_path = os.path.join(working_dir, EXECUTION_LOGFILE)
        if show_output and not is_main_thread:
            print_logfile(execution_path, logger)
            print_logfile(str(module_log_file_path), logger)
        if run:
            run.upload_file(EXECUTION_LOGFILE, execution_path)
        logger.remove_current_thread()

    return module_run_status


def update_visualizer(visualizer, node_id, status, log=None):
    if not visualizer or not node_id:
        return
    visualizer.send_message(message='status', content={node_id: status})
    if log:
        visualizer.send_message(message='logs', content={node_id: {'70_driver_log.txt': log}})


def update_module_status(module_status, status=None, run_details_url=None):
    if not status:
        status = {'startTime': None,
                  'endTime': None,
                  'runStatus': None,
                  'statusDetail': None,
                  'runDetailsUrl': run_details_url}
    status['runStatus'] = RUN_STATUS[module_status]

    if module_status == 'Running' and not status['startTime']:
        status['startTime'] = datetime.now().isoformat()
    elif module_status == 'Completed' or module_status == 'Failed':
        status['endTime'] = datetime.now().isoformat()
    return status


def check_module_run_output(module, volumes):
    # will create module output file/folder which not exists after module run
    if len(module.outputs) == 0:
        return
    path = Path([key for key, value in volumes.items() if value['bind'] == CONTAINER_OUTPUT_PATH][0])
    for output_name in module.outputs.keys():
        if not (path / output_name).exists():
            output_config = module._get_output_config_by_argument_name(output_name)
            if output_config.data_type_id == 'AnyFile':
                (path / output_name).touch()
            else:
                (path / output_name).mkdir(parents=True)


def _update_run_log_and_output(run, module, working_dir, module_run_success, module_log_file_path):
    log_file_name = module_log_file_path.name
    upload_log_file = run.upload_file(log_file_name, str(module_log_file_path))
    log_file_url = upload_log_file.artifact_content_information[log_file_name].content_uri

    # Upload output to experiment
    for output_port_name in module.outputs.keys():
        output_port_path = os.path.join(working_dir, OUTPUT_DIR_NAME, output_port_name)
        if os.path.exists(output_port_path):
            if os.path.isdir(output_port_path):
                run.upload_folder(output_port_name, output_port_path)
            else:
                run.upload_file(output_port_name, output_port_path)
    if module_run_success:
        run.complete()
    else:
        run.fail()
    print('Finish upload run status to runhistory')
    return log_file_url


def _prepare_debug_config_for_module(module, module_image, command, volumes, environment, snapshot_path):
    from azureml.pipeline.wrapper.debug._module_debug_helper import DebugLocalModuleHelper
    from azureml.pipeline.wrapper.dsl._utils import _change_working_dir
    with _change_working_dir(snapshot_path):
        mount_val = "source={},target={},type=bind,consistency=cached"
        mounts = [mount_val.format(key, value['bind'])for key, value in volumes.items()]
        DebugLocalModuleHelper.prepare_dev_container(module_image.image_name,
                                                     name=module.name,
                                                     containerEnv=environment,
                                                     mounts=mounts,
                                                     )
        DebugLocalModuleHelper.create_launch_config(module.name, command[1:])
        from azureml.pipeline.wrapper.dsl._utils import _print_step_info
        _print_step_info([
            f'Module run failed, you can debug module in vscode by steps:',
            '1. Pressing F1, click "Remote-Containers: Reopen in Container".',
            f'2. In Status Bar, selecting python interpreter path "{module_image.python_path}"',
            "3. Pressing F5 to start debugging."])


def _get_module_image_details(module):
    # Get environment of module
    env_config = json.loads(module._module_dto.module_entity.runconfig)['Environment']

    # In Environment deserialize, keys need lowercase first letter
    def trans_config_key(config):
        new_config = {}
        for key in config.keys():
            lower_key = key[:1].lower() + key[1:]
            if isinstance(config[key], dict):
                new_config[lower_key] = trans_config_key(config[key])
            else:
                new_config[lower_key] = config[key]
        return new_config

    env_config = trans_config_key(env_config)
    env = Environment._deserialize_and_add_to_object(env_config)
    env.name = module.name
    env = env.register(module.workspace)
    return _get_or_build_image(module.workspace, env)


def _get_or_build_image(workspace, env):
    detail = env.get_image_details(workspace)
    if not detail['imageExistsInRegistry']:
        build_info = env.build(workspace)
        print('Start image building')
        build_info.wait_for_completion()
        detail = env.get_image_details(workspace)

        if not detail['imageExistsInRegistry']:
            raise Exception('Build image failed, image not in registry')
        address = detail['dockerImage']['registry']['address']
        print(f'Image building success for environment {env.name}, '
              f'image name: {address}/{detail["dockerImage"]["name"]}')
    return detail


@lru_cache()
def _get_snapshot_content(module):
    service_caller = DesignerServiceCaller(module.workspace)
    snapshot_url = service_caller.get_module_snapshot_url_by_id(module_id=module._module_dto.module_version_id)
    response = requests.get(snapshot_url, allow_redirects=True)
    return response.content


def _get_module_snapshot(module, target_dir):
    content = _get_snapshot_content(module)
    # extract snapshot to script path
    with zipfile.ZipFile(BytesIO(content), 'r') as zip_ref:
        zip_ref.extractall(target_dir)


def _download_snapshot(snapshot_url, script_path):
    # download snapshot to target directory
    response = requests.get(snapshot_url, allow_redirects=True)
    # extract snapshot to script path
    with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
        zip_ref.extractall(script_path)


def _set_environment_variables_for_run(run):
    env = {
        'AZUREML_RUN_ID': run.id,
        'AZUREML_ARM_SUBSCRIPTION': run.experiment.workspace.subscription_id,
        'AZUREML_ARM_RESOURCEGROUP': run.experiment.workspace.resource_group,
        'AZUREML_ARM_WORKSPACE_NAME': run.experiment.workspace.name,
        'AZUREML_ARM_PROJECT_NAME': run.experiment.name,
        'AZUREML_RUN_TOKEN': run._client.run.get_token().token,
        'AZUREML_WORKSPACE_ID': run.experiment.workspace._workspace_id,
        'AZUREML_SERVICE_ENDPOINT': run._client.run.get_cluster_url(),
        'AZUREML_DISCOVERY_SERVICE_ENDPOINT': run.experiment.workspace.discovery_url,
    }
    return env


def _generate_command(
        module, working_dir, use_docker, remove_none_value=True, check_input_data_exist=True,
        container_input_prefix=CONTAINER_INPUT_PATH, container_output_prefix=CONTAINER_OUTPUT_PATH):
    environment = {}
    volumes = {}

    # Mount input path to container and replace input port value in arguments
    input_path = {}
    for input_name, input_item in module.inputs.items():
        input_config = module._get_input_config_by_argument_name(input_name)
        if isinstance(input_item.dset, str) or isinstance(input_item.dset, Path):
            # Change to absolute path to avoid relative import error when running locally
            input_item_path = Path(input_item.dset).resolve().absolute()
            port_name = module._pythonic_name_to_input_map[input_name]
            input_data_type = \
                next(input.data_type_ids_list for input in module._interface_inputs if input.name == port_name)
            if ['AnyFile'] == input_data_type:
                if not input_item_path.is_file():
                    input_item_path = next(filter(lambda item: item.is_file(), input_item_path.iterdir()), None)
            if not check_input_data_exist or input_item_path.exists():
                if use_docker:
                    if str(input_item_path) in volumes:
                        input_port_path = volumes[str(input_item_path)]['bind']
                    else:
                        input_port_path = container_input_prefix + '/' + \
                            os.path.basename(input_name)
                        volumes[str(input_item_path)] = {'bind': input_port_path, 'mode': 'ro'}
                else:
                    input_port_path = str(input_item_path)
                input_path[input_name] = input_port_path
            else:
                if check_input_data_exist and not input_config.is_optional:
                    raise ValueError(
                        f'Local input port path for "{input_name}" does not exist, path: {input_item.dset}')
        else:
            if not input_config.is_optional:
                raise ValueError(f'Input port "{input_name}" not set')

    # Mount output path to container and replace output port value in arguments
    output_portname_container_path = {}
    for output_port_name in module.outputs.keys():
        if use_docker:
            output_port_path = container_output_prefix + '/' + output_port_name
        else:
            output_port_path = os.path.join(working_dir, OUTPUT_DIR_NAME, output_port_name)
        output_portname_container_path[output_port_name] = output_port_path
    output_path = os.path.join(working_dir, OUTPUT_DIR_NAME)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    volumes[output_path] = {'bind': CONTAINER_OUTPUT_PATH, 'mode': 'rw'}

    # Get module run command
    command = module._get_arguments(input_path, output_portname_container_path, remove_none_value)

    return command, volumes, environment


def _prepare_module_run(node, working_dir, pipeline_parameters, module_to_node_mapping):
    from ._module import _InputBuilder
    # Replace node inputs and parameters
    copy_node = copy.deepcopy(node)
    input_path = {}
    for input_name, input_value in copy_node.inputs.items():
        input_path[input_name] = _prepare_module_inputs(
            node.workspace, input_name, input_value.dset, working_dir,
            pipeline_parameters, module_to_node_mapping)
    copy_node.set_inputs(**input_path)

    params_value = {}
    for param_name, param_value in copy_node._parameter_params.items():
        if isinstance(param_value, PipelineParameter):
            params_value[param_name] = get_pipeline_param(param_name, param_value, pipeline_parameters)
        elif isinstance(param_value, _InputBuilder):
            params_value[param_name] = param_value.dset.default_value
    copy_node.set_parameters(**params_value)
    return copy_node


def _prepare_module_inputs(workspace, input_name, dset, working_dir,
                           pipeline_parameters, module_to_node_mapping):
    # Download dataset and replace node inputs to local data path
    from ._pipeline_run_orchestrator import WORKING_DIR
    from ._module import _OutputBuilder, _InputBuilder
    if isinstance(dset, _InputBuilder):
        return _prepare_module_inputs(
            workspace, input_name, dset.dset, working_dir,
            pipeline_parameters, module_to_node_mapping)
    if isinstance(dset, _OutputBuilder):
        return os.path.join(module_to_node_mapping[dset.module_instance_id][WORKING_DIR], OUTPUT_DIR_NAME, dset._name)
    elif isinstance(dset, DataReference) or isinstance(dset, FileDataset) or \
            isinstance(dset, DataPath) or isinstance(dset, DatasetConsumptionConfig):
        return _download_input_data(workspace, input_name, dset, working_dir)
    elif isinstance(dset, PipelineParameter):
        default_value = dset.default_value if not pipeline_parameters or \
            (input_name not in pipeline_parameters.keys()) else pipeline_parameters[input_name]
        return _download_input_data(workspace, input_name, default_value, working_dir)
    elif isinstance(dset, str) or not dset:
        return dset
    else:
        raise ValueError(f"Unknown type {type(dset)} for node input dataset {input_name}")


def _download_input_data(workspace, input_name, dset, working_dir):
    # Download module input dataset to local
    if isinstance(dset, DataReference):
        data_store_name = dset.data_store_name
        path_on_data_store = dset.path_on_datastore
        blob_data_store = Datastore.get(workspace, data_store_name)
        target_path = Path(working_dir) / path_on_data_store
        if target_path.exists():
            return str(target_path)
        blob_data_store.download(
            target_path=working_dir, prefix=path_on_data_store, overwrite=False)
        target_path.mkdir(exist_ok=True, parents=True)
        return str(target_path)
    elif isinstance(dset, FileDataset):
        dataset_id = dset.id
        dataset_name = dset.name
        target_path = Path(working_dir, dataset_name if dataset_name else dataset_id)
        if target_path.exists():
            return str(target_path)
        dataset = Dataset.get_by_id(workspace, dataset_id)
        dataset.download(target_path=str(target_path), overwrite=False)
        return str(target_path)
    elif isinstance(dset, DataPath):
        path_on_data_store = dset._path_on_datastore
        target_path = Path(working_dir) / path_on_data_store
        if target_path.exists():
            return str(target_path)
        dset._datastore.download(
            target_path=working_dir, prefix=path_on_data_store, overwrite=False)
        target_path.mkdir(exist_ok=True, parents=True)
        return str(target_path)
    elif isinstance(dset, DatasetConsumptionConfig):
        return _download_input_data(workspace, input_name, dset.dataset, working_dir)
    else:
        raise ValueError('Input dataset is of unsupported type: {0}'.format(type(dset).__name__))


def get_pipeline_param(param_name, param_value, pipeline_parameters):
    default_value = pipeline_parameters[param_value.name] if pipeline_parameters and \
        param_value.name in pipeline_parameters.keys() else param_value.default_value
    if isinstance(default_value, int) or isinstance(default_value, str) or \
            isinstance(default_value, bool) or isinstance(default_value, float):
        return default_value
    else:
        raise ValueError('Node parameter is of unsupported type: {0}'.format(type(default_value).__name__))


def _run_docker_container(docker_client, image_location, command=None,
                          volumes=None, environment=None):
    is_wsl_or_container = is_in_container() or is_in_wsl1()
    if is_wsl_or_container:
        container = docker_client.containers.create(
            image_location, working_dir=CONTAINER_MOUNT_SCRIPTS_PATH, environment=environment,
            stdin_open=True, privileged=True, tty=True)
    else:
        container = docker_client.containers.create(
            image_location, working_dir=CONTAINER_MOUNT_SCRIPTS_PATH, environment=environment,
            volumes=volumes, stdin_open=True, privileged=True, tty=True)
    try:
        container.start()
        if is_wsl_or_container:
            command_result, stdout = exec_command_in_wsl1_container(container, command, volumes)
        else:
            command_result, stdout = container.exec_run(command)
        stdout = stdout.decode()
        if command_result != 0:
            return container, False, stdout
    except Exception:
        return container, False, stdout
    finally:
        container.stop()
    return docker_client.containers.get(container.id), True, stdout


def _copy_from_docker(container, source, target):
    try:
        data_stream, _ = container.get_archive(source)
        tar_file = target + '.tar'
        with open(tar_file, 'wb') as f:
            for chunk in data_stream:
                f.write(chunk)
        with tarfile.open(tar_file, mode='r') as tar:
            for file_name in tar.getnames():
                tar.extract(file_name, os.path.dirname(target))
    except Exception as e:
        raise RuntimeError(e)
    finally:
        os.remove(tar_file)


def is_in_container():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )


def is_in_wsl1():
    process = subprocess.run("systemd-detect-virt -c", shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
    return 'wsl' in process.stdout


def exec_command_in_wsl1_container(container, command, volumes):
    """
        In WSL1 and container, will execute docker command in host machine, so folder in WSL1/container
        cannot mount in docker container. Using docker cp to replace mounting.
        :param container: container
        :type container: docker.container
        :param command: execute command in container
        :type command: list
        :param volumes: volumes need to mount in container
        :type volumes: dict
        :return command_result: command run result, if not 0, may some error when execute
                stdout: log of executing command
        :rtype int, bytes
    """
    print('Warning: Running in WSL1 or container')
    # copy code and data to container
    for key, item in volumes.items():
        if not os.path.exists(key):
            continue
        if os.path.isdir(key):
            write_dir_in_container(container, item['bind'], key)
        else:
            with open(key, 'rb') as f:
                write_file_in_container(container, item['bind'], f.read())

    # execute command
    command_result, stdout = container.exec_run(command)

    # copy reuslt to local
    for key, item in volumes.items():
        if item['bind'].startswith(CONTAINER_OUTPUT_PATH):
            _copy_from_docker(container, item['bind'], key)
    return command_result, stdout


def print_logfile(log_path, logger):
    if os.path.exists(log_path):
        print_str = f"\n{log_path}\n{'=' * len(log_path)}\n"
        with open(log_path) as f:
            for line in f.readlines():
                print_str += line
        logger._terminal.write(print_str)
        logger._terminal.flush()


class Logger(object):
    _instance_lock = Lock()
    tid_to_logfile = {}
    _instance = None

    def __new__(cls, *args, **kwargs):
        with Logger._instance_lock:
            if cls._instance is None:
                cls._instance = super(Logger, cls).__new__(Logger)
                cls._terminal = sys.stdout
        return cls._instance

    def __init__(self, log_path, show_terminal=True):
        self.tid_to_logfile[currentThread().ident] = open(log_path, "a")
        self._show_terminal = show_terminal
        if sys.stdout != self:
            sys.stdout = self

    def write(self, message):
        if self._show_terminal:
            self._terminal.write(message)
            self._terminal.flush()
        if currentThread().ident in self.tid_to_logfile.keys():
            self.tid_to_logfile[currentThread().ident].write(message)
            self.tid_to_logfile[currentThread().ident].flush()

    def flush(self):
        if self._show_terminal:
            self._terminal.flush()
        if currentThread().ident in self.tid_to_logfile.keys():
            self.tid_to_logfile[currentThread().ident].flush()

    def remove_current_thread(self):
        log_file = self.tid_to_logfile.pop(currentThread().ident, None)
        if log_file:
            log_file.close()
        if len(self.tid_to_logfile) == 0:
            sys.stdout = self._terminal
