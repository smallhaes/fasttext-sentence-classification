# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from .._restclients.designer.models.sub_pipeline_definition_py3 import SubPipelineDefinition


class _PipelineDefinitionStack:
    """ A stack stores all :class`~designer.models.SubPipelineDefinition`
    in creating state created by :class`azureml.pipeline.wrapper.dsl.pipeline`
    """

    def __init__(self):
        self.items = []

    def top(self) -> str:
        return self.items[-1]

    def pop(self) -> str:
        return self.items.pop()

    def push(self, item: SubPipelineDefinition):
        error_msg = "_PipelineDefinitionStack only allows pushing `SubPipelineDefinition` element"
        assert isinstance(item, SubPipelineDefinition), error_msg
        return self.items.append(item)

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


_pipeline_definition_stack = _PipelineDefinitionStack()
