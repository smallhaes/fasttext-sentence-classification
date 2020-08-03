# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml.core import Workspace, Datastore
from azureml.data.data_reference import DataReference
from ._restclients.service_caller import DesignerServiceCaller
from ._utils import _sanitize_python_variable_name

_cached_global_dataset = None


class _GlobalDataset(DataReference):
    """
    A class that represents a global dataset provided by AzureML.
    """
    def __init__(self, workspace: Workspace, data_store_name: str, relative_path: str):
        self.data_store_name = data_store_name
        self.relative_path = relative_path
        super().__init__(datastore=Datastore(workspace, name=data_store_name),
                         data_reference_name=_sanitize_python_variable_name(relative_path),
                         path_on_datastore=relative_path)


def get_global_dataset_by_path(workspace: Workspace, name, path):
    """
    Retrieve global dataset provided by AzureML using name and path.

    This is intended only for internal usage and *NOT* part of the public APIs.
    Please be advised to not rely on its current behaviour.
    """
    service_caller = DesignerServiceCaller(workspace)
    global _cached_global_dataset
    if _cached_global_dataset is None:
        all_dataset = service_caller.list_datasets()
        _cached_global_dataset = [d for d in all_dataset if d.aml_data_store_name == 'azureml_globaldatasets']
    dataset = next((x for x in _cached_global_dataset if x.relative_path == path), None)
    if dataset is None:
        raise ValueError(f'dataset not found with path: {path}')
    return _GlobalDataset(workspace, dataset.aml_data_store_name, dataset.relative_path)
