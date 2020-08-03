# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml.pipeline.wrapper._module_run_helper import _get_module_image_details
from azureml._model_management._util import pull_docker_image
from azureml._model_management._util import get_docker_client


class ImageBase:
    def __init__(self, image_details):
        self.registry = image_details['dockerImage']['registry']
        if not self.registry['address']:
            self.image_name = image_details['dockerImage']['name']
        else:
            self.image_name = '%s/%s' % (self.registry['address'], image_details['dockerImage']['name'])
        self.python_path = image_details['pythonEnvironment']['interpreterPath']

    def pull_module_image(self):
        # pull image
        docker_client = get_docker_client()
        pull_docker_image(
            docker_client, self.image_name, self.registry['username'], self.registry['password'])


class ModuleImage(ImageBase):
    def __init__(self, module):
        image_detail = _get_module_image_details(module)
        super(ModuleImage, self).__init__(image_detail)
        self.module = module
