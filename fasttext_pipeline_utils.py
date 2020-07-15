import os
from azureml.core import Workspace, Dataset
from azureml.pipeline.core.graph import DataType
from azureml.pipeline.wrapper import Module, dsl
from azureml.core.compute import AmlCompute, AksCompute, ComputeTarget
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.model import Model, InferenceConfig
from azureml.core.datastore import Datastore
from azureml.core.environment import Environment
from azureml.core.webservice import LocalWebservice, AciWebservice, Webservice, AksWebservice


def choose_workspace(subscription_id, resource_group, workspace_name, tenant_id):
    interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id)
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    print('name:', workspace.name)
    print('resource_group', workspace.resource_group)
    print('location', workspace.location)
    print('subscription_id', workspace.subscription_id)
    print('compute_targets', workspace.compute_targets.keys())
    return workspace


# choose compute target
def choose_compute_target(workspace, name):
    try:
        aml_compute = AmlCompute(workspace, name)
        print("Found existing compute target: {}".format(name))
    except:
        print("Creating new compute target: {}".format(name))

        provisioning_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
                                                                    min_nodes=1,
                                                                    max_nodes=4)
        aml_compute = ComputeTarget.create(workspace, name, provisioning_config)
        aml_compute.wait_for_completion(show_output=True)
    print(aml_compute)
    return aml_compute


# register your own datatype
def register_datatype(workspace, name, description, is_directory):
    # won't register repeatedly
    DataType.create_data_type(workspace=workspace, name=name, description=description, is_directory=is_directory)
    print('Datatype of {} is registered'.format(name))


# load data
def load_dataset(workspace, name, path, description):
    if name not in workspace.datasets:
        print('Registering the dataset of {}...'.format(name))
        data = Dataset.File.from_files(path=path)
        data.register(name=name, description=description, workspace=workspace)
    print('Successfully loaded {}'.format(name))
    return workspace.datasets[name]


# load module
def load_module(workspace, namespace, name, yaml_file_path):
    try:
        module_func = Module.load(workspace=workspace, namespace=namespace, name=name)
        print('found the module of {}'.format(name))
        return module_func
    except:
        print('not found the module of {}, register it now...'.format(name))
        module_func = Module.register(workspace=workspace, yaml_file=yaml_file_path)
        return module_func


# register the trained model from child run
# This does not work right now, because it will upload the zip file in Outputs+logs ranther than the real model
def register_model_from_child_run(child_run, model_name, model_path, tags=None):
    model = child_run.register_model(model_name=model_name, model_path=model_path, tags=tags)


# pay attention to the situation of reuse
def get_source_child_run_id(child_run):
    properties = child_run.properties
    if 'azureml.reusedrunid' in properties:
        return properties['azureml.reusedrunid']
    else:
        return child_run.id


# down model from blob
# whether the downloading process is successful
def download_model(workspace, path_on_data_store, target_path='.', overwrite=True):
    blob_data_store = Datastore.get_default(workspace)
    number_of_files_successfully_downloaded = blob_data_store.download(target_path=target_path,
                                                                       prefix=path_on_data_store, overwrite=overwrite)
    if number_of_files_successfully_downloaded == 0:
        print('there is no model downloaded')
    else:
        print('model is downloaded to the directory of {}'.format(target_path))


# register the trained model from local
def register_model_from_local(workspace, model_name, model_path, tags=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = Model.register(workspace=workspace, model_name=model_name, model_path=model_path, tags=tags)
    print('model is registered from local')
    return model


# register env
# won't increase the version automatically
def register_enviroment(workspace, name, file_path):
    myenv = Environment.from_conda_specification(name=name, file_path=file_path)
    myenv = myenv.register(workspace=workspace)
    return myenv


# get Environment object
def get_env(workspace, name, version='1'):
    env = Environment.get(workspace=workspace, name=name, version=version)
    return env


# define inference configuration
# it will upload all files in the source_diretory
def define_inference_configuration(entry_script, source_directory, environment):
    inference_config = InferenceConfig(entry_script=entry_script, source_directory=source_directory,
                                       environment=environment)
    return inference_config


# get model from workspace
def get_model_from_workspace(workspace, model_name, version):
    model = Model.get_model_path(model_name=model_name, version=version, _workspace=workspace)
    return model


# a machine learning model deployed as a local web service endpoint
# benefit of deploying locally: we could use the same "service_name"
def deploy_locally(workspace, service_name, models, inference_config, port=8890):
    deployment_config = LocalWebservice.deploy_configuration(port=port)
    service = Model.deploy(workspace=workspace, name=service_name, models=models, inference_config=inference_config,
                           deployment_config=deployment_config)
    service.wait_for_deployment(show_output=True)
    print(service.state)
    return service


# deploy a model to Azure Container Instances
# we could not use the same "service_name"
def deploy_to_ACI(workspace, service_name, models, inference_config, cpu_cores=1, memory_gb=1):
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=cpu_cores, memory_gb=memory_gb)
    service = Model.deploy(workspace, service_name, models=models, inference_config=inference_config,
                           deployment_config=deployment_config)
    service.wait_for_deployment(show_output=True)
    print(service.state)
    return service


# attach to AKS
# create a attachment for a specific AKS
def attach_to_AKS(workspace, attachment_name, resource_id, cluster_purpose=None):
    attach_config = AksCompute.attach_configuration(resource_id=resource_id, cluster_purpose=cluster_purpose)
    aks_target = ComputeTarget.attach(workspace, attachment_name, attach_config)
    return aks_target


# deploy a model to Azure Kubernetes Service
# we could not use the same "service_name"
def deploy_to_AKS(workspace, attachment_name, service_name, models, inference_config, token_auth_enabled=True,
                  cpu_cores=1, memory_gb=1):
    aks_target = AksCompute(workspace, attachment_name)
    # If deploying to a cluster configured for dev/test, ensure that it was created with enough
    # cores and memory to handle this deployment configuration. Note that memory is also used by
    # things such as dependencies and AML components.
    deployment_config = AksWebservice.deploy_configuration(cpu_cores=cpu_cores, memory_gb=memory_gb,
                                                           token_auth_enabled=token_auth_enabled)
    service = Model.deploy(workspace, service_name, models, inference_config, deployment_config, aks_target)
    service.wait_for_deployment(show_output=True)
    print(service.state)
    return service
