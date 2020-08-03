# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This package contains classes used to manage Compute Targets objects within
Azure Machine Learning. Keep the files here is a temporary solution"""
from azureml._base_sdk_common import __version__ as VERSION
from .k8scompute import AksCompute
from .cmakscompute import CmAksCompute

__version__ = VERSION

__all__ = [
    'AksCompute',
    'CmAksCompute'
]
