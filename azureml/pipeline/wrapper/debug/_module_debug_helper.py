# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import json
import platform
from pathlib import Path

from ._step_run_debug_helper import DebugStepRunHelper
from azureml.pipeline.wrapper.dsl._utils import _print_step_info
from azureml.pipeline.wrapper.dsl._utils import BackUpFiles
from azureml.pipeline.wrapper.dsl._utils import FileExistProcessor
from azureml.pipeline.wrapper.debug._constants import VSCODE_DIR, INPUT_DIR, LAUNCH_CONFIG, \
    CONTAINER_DIR, GIT_IGNORE, CONTAINER_CONFIG


class DebugLocalModuleHelper(DebugStepRunHelper):

    @staticmethod
    def prepare_dev_container(image_name, **kwargs):
        # create container config
        data = {'image': image_name}
        for key, val in kwargs.items():
            data[key] = val
        target = os.getcwd()
        container_config_path = os.path.join(CONTAINER_DIR, CONTAINER_CONFIG)
        if os.path.exists(container_config_path):
            with open(container_config_path, 'r') as f:
                container_config = json.load(f)
                if all((key in container_config.keys() and container_config[key] == value)
                       for key, value in data.items()):
                    return
        with BackUpFiles(target) as backup_folder:
            file_exist_processor = FileExistProcessor(
                Path(target), True, backup_folder)
            file_exist_processor.process_or_skip(container_config_path,
                                                 DebugLocalModuleHelper.create_container_config,
                                                 **data
                                                 )

    @staticmethod
    def create_launch_config(config_name, arguments):
        """create launch config for module, used for module debug.
           If workspace/.vscode/launch.json not exist, will create config file. If it exist,
           will only overwrite config which name equals config_name.

        :param config_name: debug config name
        :type config_name: str
        :param arguments: module run arguments
        :type arguments: list
        :return launch config path
        :rtype str
        """
        # create python debug config
        with open(INPUT_DIR / LAUNCH_CONFIG) as container_config:
            data = json.load(container_config)
        config = {
            "name": config_name,
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
        # module debug config
        if '-m' == arguments[0]:
            config['module'] = arguments[1]
            arguments = arguments[2:]
        else:
            config['program'] = '${workspaceFolder}/' + arguments[0]
            arguments = arguments[1:]

        config['linux'] = {
            "args": arguments
        }
        data['configurations'].append(config)

        os.makedirs(VSCODE_DIR, exist_ok=True)
        launch_config_path = os.path.join(VSCODE_DIR, LAUNCH_CONFIG)

        def write_launch_file(data, launch_config_path):
            # if exist launch.json in target, add container config in it.
            if os.path.exists(launch_config_path):
                with open(launch_config_path, 'r') as config_file:
                    exist_config = json.load(config_file)

                    # check module debug config exists in launch.json
                    exist_container_config = next(
                        filter(
                            lambda launch_config: launch_config[1]['name'] == data['configurations'][0]['name'],
                            enumerate(exist_config['configurations'])
                        ),
                        None
                    )

                    # In launch.json, args will exits in config[args] of config[os_type][args].
                    # If module debug config in launch.json and user os is windows,
                    # will write current args in config[windows], user can locally debug using same config.
                    # If module debug config exists and user os is linux, remote config args will replace current args.
                    if exist_container_config:
                        if platform.system() == 'Windows':
                            if 'args' in exist_container_config[1]:
                                config['windows'] = {'args': exist_container_config[1]['args']}
                                exist_container_config[1].pop('args')
                            elif 'windows' in exist_container_config[1]:
                                config['windows'] = exist_container_config[1]['windows']
                        exist_config['configurations'][exist_container_config[0]].update(data['configurations'][0])
                    else:
                        if not exist_config['configurations']:
                            exist_config['configurations'] = []
                        exist_config['configurations'].append(data['configurations'][0])
                    data = exist_config
            with open(launch_config_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)

        target = os.getcwd()
        with BackUpFiles(target) as backup_folder:
            file_exist_processor = FileExistProcessor(Path(target), True, backup_folder)
            file_exist_processor.process_or_skip(launch_config_path, write_launch_file, data, launch_config_path)
        _print_step_info(f'Created launch config "{config_name}" for module.')
        return launch_config_path

    @staticmethod
    def update_git_ignore_file(file_name):
        with open(GIT_IGNORE, 'a+') as f:
            lines = f.readlines()
            for line in lines:
                if line == file_name:
                    return
            f.write('\n' + file_name)
