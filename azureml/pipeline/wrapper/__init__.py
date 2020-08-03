# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains core functionality for Azure Machine Learning pipelines, which are configurable machine learning workflows.

Azure Machine Learning pipelines allow you to create resusable machine learning workflows that can be used as a
template for your machine learning scenarios. This package contains the core functionality for working with
Azure ML pipelines.

A machine learning pipeline is represented by a :class:`azureml.pipeline.wrapper.Pipeline` object which can be
constructed for a collection of :class:`azureml.pipeline.wrapper.Module` object that can sequenced and parallelized,
or be created with explicit dependencies.

You can create and work with pipelines in a Jupyter Notebook or any other IDE with the Azure ML SDK installed.
"""

from ._module import Module
from ._pipeline import Pipeline
from .pipeline_run import PipelineRun, StepRun, InputPort, OutputPort

__all__ = [
    'Module',
    'Pipeline',
    'PipelineRun',
    'StepRun',
    'InputPort',
    'OutputPort'
]
