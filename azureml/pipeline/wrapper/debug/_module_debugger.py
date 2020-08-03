# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
from pathlib import Path

from azureml.pipeline.wrapper.debug._image import ModuleImage
from azureml.pipeline.wrapper.debug._module_debug_helper import DebugLocalModuleHelper
from azureml.pipeline.wrapper.debug._step_run_debugger import DebugRunner
from azureml.core import Workspace
from azureml.pipeline.wrapper.dsl._module_local_param_builder import _ModuleLocalParamBuilderFromModule
from azureml.pipeline.wrapper.dsl._utils import _print_step_info, _change_working_dir
from azureml.pipeline.wrapper._module_run_helper import _generate_command
from azureml.pipeline.wrapper._module import Module


class LocalModuleDebugger(DebugRunner):
    def __init__(self,
                 workspace_name=None,
                 resource_group=None,
                 subscription_id=None,
                 yaml_file=None):
        if None in [yaml_file, workspace_name, resource_group, subscription_id]:
            raise ValueError(
                'yaml_file, workspace_name, resource_group, subscription_id cannot be null.')
        target = str(Path(yaml_file).parent)
        super().__init__(target=target)

        _print_step_info(f'Preparing module remote debug config')

        self.workspace = Workspace(subscription_id=subscription_id, resource_group=resource_group,
                                   workspace_name=workspace_name)
        module_func = Module.from_yaml(self.workspace, yaml_file)
        self.module = module_func()

        self.module_param_builder = _ModuleLocalParamBuilderFromModule(self.module, Path(self.target))
        self.module_param_builder.build()

        self.run = self.run_step(self.local_setup)

    def local_setup(self):
        with _change_working_dir(self.target):

            module_image = ModuleImage(self.module)
            module_image.pull_module_image()
            self.python_path = module_image.python_path

            workspace_folder = "/workspace/{}".format(os.path.basename(self.target))
            workspace_mount = "source=${localWorkspaceFolder}," + \
                              f"target={workspace_folder},type=bind,consistency=delegated"

            container_input_prefix = str(self.module_param_builder.input_dir)
            container_output_prefix = str(self.module_param_builder.output_dir)
            # generate arguments
            command, volumes, _ = _generate_command(
                self.module, self.target, True, remove_none_value=False, check_input_data_exist=False,
                container_input_prefix=container_input_prefix, container_output_prefix=container_output_prefix)

            DebugLocalModuleHelper.prepare_dev_container(module_image.image_name,
                                                         name=self.module.name,
                                                         containerEnv={},
                                                         workspaceMount=workspace_mount,
                                                         workspaceFolder=workspace_folder
                                                         )
            DebugLocalModuleHelper.create_launch_config(self.module_param_builder.module_file_name,
                                                        command[1:]
                                                        )
