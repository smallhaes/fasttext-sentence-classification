# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from pathlib import PurePosixPath, PureWindowsPath, Path

REMOTE_MOUNT_DIR = PurePosixPath('local_mount_dir')
LOCAL_MOUNT_DIR = PureWindowsPath('local_mount_dir')
CONDA_FILE_NAME = 'conda.yaml'
DOCKER_FILE_NAME = 'Dockerfile'
DATA_REF_PREFIX = '$AZUREML_DATAREFERENCE_'

SUBSCRIPTION_KEY = 'subscriptions'
RESOURCE_GROUP_KEY = 'resourcegroups'
WORKSPACE_KEY = 'workspaces'
EXPERIMENT_KEY = 'experiments'
ID_KEY = 'id'
DRAFT_KEY = 'Normal'
RUN_KEY = 'runs'

INPUT_DIR = Path(__file__).parent.parent / 'dsl' / 'data' / 'inputs'
VSCODE_DIR = '.vscode'
CONTAINER_DIR = '.devcontainer'
LAUNCH_CONFIG = 'launch.json'
CONTAINER_CONFIG = 'devcontainer.json'
OUTPUT_DIR = 'outputs'
DIR_PATTERN = r'[^a-zA-Z0-9]'
DEBUG_FOLDER = 'data'
INPUTS_FOLDER = 'inputs'
OUTPUTS_FOLDER = 'outputs'
GIT_IGNORE = '.gitignore'

DOCKERFILE_TEMPLATE = '''
FROM {image_name}
ADD conda.yaml conda.yaml
RUN conda env create -f conda.yaml
'''
