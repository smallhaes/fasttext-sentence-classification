# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import re
import os
import sys

from functools import wraps

from azureml.pipeline.wrapper._restclients.service_caller import DesignerServiceCaller
from azureml.pipeline.wrapper.debug._constants import DIR_PATTERN
from azureml.pipeline.wrapper.debug._step_run_debug_helper import DebugOnlineStepRunHelper, _print_step_info, logger
from azureml.pipeline.wrapper.dsl._utils import _change_working_dir


class DebugRunner:
    def __init__(self, target=None):
        if target is None:
            target = os.getcwd()
        self.target = target
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.step_run = None
        self.step_detail = None
        self.python_path = None

    def run(self):
        pass

    def run_step(self, func):
        # run all steps passed in failed_steps
        @wraps(func)
        def wrapper():
            # Hint to install requirements
            DebugOnlineStepRunHelper.installed_requirements()

            func()

        return wrapper


class OnlineStepRunDebugger(DebugRunner):
    def __init__(self,
                 url=None,
                 run_id=None,
                 experiment_name=None,
                 workspace_name=None,
                 resource_group_name=None,
                 subscription_id=None,
                 target=None,
                 dry_run=False):
        super().__init__(target=target)

        if url is not None:
            run_id, experiment_name, workspace_name, resource_group_name, subscription_id = \
                DebugOnlineStepRunHelper.parse_designer_url(url)
        if all(var is not None for var in
               [run_id, experiment_name, workspace_name, resource_group_name, subscription_id]):
            self.run_id = run_id
            self.experiment_name = experiment_name
            self.workspace_name = workspace_name
            self.resource_group_name = resource_group_name
            self.subscription_id = subscription_id
            self.dry_run = dry_run
            self.step_run = None
            self.step_detail = None
            self.workspace = None
            self.service_caller = None
            self.step_id = None
        else:
            raise ValueError(
                'One of url or step run params(run_id, experiment_name, '
                'workspace_name, resource_group_name, subscription_id) should be passed.')
        # manually call decorator passing self to decorator
        self.run = self.run_step(self.remote_setup)

    def remote_setup(self):
        # won't pull image and download data for test
        if '_TEST_ENV' in os.environ:
            logger.warning("Envrionment variable _TEST_ENV is set, won't pull image and download data.")
            self.dry_run = True
        _print_step_info(f'Fetching pipeline step run metadata')
        self.step_run = DebugOnlineStepRunHelper.get_pipeline_run(self.run_id, self.experiment_name,
                                                                  self.workspace_name, self.resource_group_name,
                                                                  self.subscription_id)
        self.step_detail = self.step_run.get_details()
        step_id = '%s:%s' % (self.step_run.name, self.step_run.id)
        step_id = re.sub(DIR_PATTERN, '_', step_id)
        self.step_id = step_id
        self.workspace = self.step_run.experiment.workspace
        self.service_caller = DesignerServiceCaller(self.workspace)

        with _change_working_dir(self.step_id):
            # prepare container and it's config
            step_run_image = DebugOnlineStepRunHelper.prepare_dev_container(self.workspace, self.step_run,
                                                                            dry_run=self.dry_run)
            self.python_path = step_run_image.python_path
            # download snapshot
            DebugOnlineStepRunHelper.download_snapshot(self.service_caller, self.step_run.parent.id, self.step_run.id,
                                                       dry_run=self.dry_run)
            # download input/output data
            port_arg_map, partial_success = DebugOnlineStepRunHelper.prepare_inputs(self.workspace, self.step_detail,
                                                                                    dry_run=self.dry_run)
            # get run arguments
            script, arguments = DebugOnlineStepRunHelper.prepare_arguments(self.step_id, self.step_detail,
                                                                           port_arg_map)
            # create vs code debug env
            DebugOnlineStepRunHelper.create_launch_config(self.step_id, self.python_path,
                                                          ['${workspaceFolder}/' + script], arguments)

            # Hint to install vscode extensions
            vscode_extensions = 'ms-vscode-remote.vscode-remote-extensionpack, ms-python.python'
            _print_step_info(
                ['Please make sure the following vscode extensions are installed: {}'.format(vscode_extensions),
                 'Open vscode in created debug workspace: {}'.format(os.getcwd()),
                 "Run 'Remote-Containers: Reopen in Container' command in VS Code to reopen the workspace in "
                 f'container, select python interpretet path "{self.python_path}" in Status Bar'
                 " and press F5 to start debugging."])

            if partial_success:
                raise RuntimeError('Dataset preparation failed, please prepare failed dataset before debugging.')


def _entry(argv):
    """CLI tool for module creating."""

    def _literal_boolean(s):
        return s.lower() != 'false'

    parser = argparse.ArgumentParser(
        prog="python -m azureml.pipeline.wrapper.debug._step_run_debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""A CLI tool for module debugging.""",
    )

    subparsers = parser.add_subparsers()

    # create module folder parser
    debug_parser = subparsers.add_parser(
        'debug',
        description='A CLI tool for online module debugging.'
    )

    debug_parser.add_argument(
        '--subscription_id', '-s', type=str,
        help="Subscription id."
    )
    debug_parser.add_argument(
        '--resource_group_name', '-r', type=str,
        help="Resource group."
    )
    debug_parser.add_argument(
        '--workspace_name', '-w', type=str,
        help="Workspace name."
    )
    debug_parser.add_argument(
        '--experiment_name', '-e', type=str,
        help="Experiment name."
    )
    debug_parser.add_argument(
        '--run_id', "-i", type=str,
        help="Run id for specific module run."
    )
    debug_parser.add_argument(
        '--target', type=str,
        help="Target directory to build environment, will use current working directory if not specified."
    )
    debug_parser.add_argument(
        "--url", type=str,
        help="Step run url."
    )
    debug_parser.add_argument(
        "--dry_run", type=_literal_boolean,
        help="Dry run."
    )

    args, _ = parser.parse_known_args(argv)

    params = vars(args)

    def _to_vars(url=None, run_id=None, experiment_name=None, workspace_name=None, resource_group_name=None,
                 subscription_id=None, target=None, dry_run=False):
        return url, run_id, experiment_name, workspace_name, resource_group_name, subscription_id, target, dry_run

    url, run_id, experiment_name, workspace_name, resource_group_name, subscription_id, target, dry_run = _to_vars(
        **params)
    if url is not None:
        debugger = OnlineStepRunDebugger(url=url, target=target, dry_run=dry_run)
    elif all(var is not None for var in
             [run_id, experiment_name, workspace_name, resource_group_name, subscription_id]):
        debugger = OnlineStepRunDebugger(run_id=run_id,
                                         experiment_name=experiment_name,
                                         workspace_name=workspace_name,
                                         resource_group_name=resource_group_name,
                                         subscription_id=subscription_id,
                                         target=target,
                                         dry_run=dry_run)
    else:
        raise RuntimeError(
            'One of url or step run params(run_id, experiment_name, '
            'workspace_name, resource_group_name, subscription_id) should be passed.')
    debugger.run()


def main():
    """Use as a CLI entry function to use OnlineStepRunDebugger."""
    _entry(sys.argv[1:])


if __name__ == '__main__':
    main()
