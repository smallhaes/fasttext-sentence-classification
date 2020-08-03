# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from .. import Pipeline


class _PipelineStack:
    """ A stack stores all :class`azureml.pipeline.wrapper.pipeline`
    in creating state created by :class`azureml.pipeline.wrapper.dsl.pipeline`

    """

    def __init__(self):
        self.items = []

    def top(self) -> Pipeline:
        return self.items[-1]

    def pop(self) -> Pipeline:
        return self.items.pop()

    def push(self, item: Pipeline):
        error_msg = "_PipelineStack only allows pushing `azureml.pipeline.wrapper.pipeline` element"
        assert isinstance(item, Pipeline), error_msg
        return self.items.append(item)

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


_pipeline_stack = _PipelineStack()
