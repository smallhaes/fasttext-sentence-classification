# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml.core import Dataset
from azureml.data.datapath import DataPath
from azureml.data.file_dataset import FileDataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig


class PipelineParameter(object):
    """Defines a parameter in a pipeline execution.

    Use PipelineParameters to construct versatile Pipelines which can be resubmitted later with varying
    parameter values. Note that we do not expose this as part of our public API yet. This is only intended for
    internal usage.

    :param name: The name of the pipeline parameter.
    :type name: str
    :param default_value: The default value of the pipeline parameter.
    :type default_value: literal values
    """
    def __init__(self, name, default_value):
        self.name = name
        self.default_value = default_value

        if not isinstance(default_value, int) and not isinstance(default_value, str) and \
            not isinstance(default_value, bool) and not isinstance(default_value, float) \
                and not isinstance(default_value, DataPath) and not isinstance(default_value, Dataset) \
                and not isinstance(default_value, DatasetConsumptionConfig) \
                and not isinstance(default_value, FileDataset) \
                and default_value is not None:
            raise ValueError('Default value is of unsupported type: {0}'.format(type(default_value).__name__))

    def __str__(self):
        """
        __str__ override.

        :return: The string representation of the PipelineParameter.
        :rtype: str
        """
        return "PipelineParameter(name={0}, default_value={1})".format(self.name, self.default_value)

    def _serialize_to_dict(self):
        if isinstance(self.default_value, DataPath):
            return {"type": "datapath",
                    "default": self.default_value._serialize_to_dict()}
        else:
            param_type = "string"
            if isinstance(self.default_value, int):
                param_type = "int"
            if isinstance(self.default_value, float):
                param_type = "float"
            if isinstance(self.default_value, bool):
                param_type = "bool"
            return {"type": param_type,
                    "default": self.default_value}
