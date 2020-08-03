from .designer.models import PipelineType, PipelineDraftMode
from azureml._html.utilities import to_html
from collections import OrderedDict
import enum


class PipelineDraft(object):
    """This is a wrapper of `azureml.pipeline.wrapper._restclients.designer.models.PipelineDraft`.

    It is intended to add more functionalities.
    """

    def __init__(self, raw_pipeline_draft, subscription_id, resource_group, workspace_name):
        self._raw_pipeline_draft = raw_pipeline_draft
        self.id = raw_pipeline_draft.id
        self.name = raw_pipeline_draft.name
        self.tags = raw_pipeline_draft.tags
        # self.properties = raw_pipeline_draft.properties
        self.subcription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name

    def _repr_html_(self):
        info = self._get_base_info_dict()
        return to_html(info)

    def _get_base_info_dict(self):
        return OrderedDict([
            ('Name', self._raw_pipeline_draft.name),
            ('Id', self._raw_pipeline_draft.id),
            ('Details page', self.get_portal_url()),
            ('Pipeline type', PipelineUtilities.convert_pipeline_type_to_str(
                self._raw_pipeline_draft.pipeline_type)),
            ('Updated on', self._raw_pipeline_draft.last_modified_date.astimezone(
            ).strftime('%B %d, %Y %I:%M %p')),
            ('Created by', self._raw_pipeline_draft.created_by),
            ('Tags', ['{}: {}'.format(tag, self.tags[tag]) for tag in self.tags]),
        ])

    def get_portal_url(self):
        netloc = ('https://ml.azure.com/visualinterface'
                  '/authoring/Normal/{}?wsid=/subscriptions/{}/resourcegroups/{}/workspaces/{}')
        return netloc.format(
            self._raw_pipeline_draft.id, self.subcription_id, self.resource_group, self.workspace_name)


class PipelineUtilities(object):
    _pipeline_type_dict = {
        '0': PipelineType.training_pipeline,
        '1': PipelineType.real_time_inference_pipeline,
        '2': PipelineType.batch_inference_pipeline,
        '3': PipelineType.unknown
    }
    _pipeline_draft_mode_dict = {
        '0': PipelineDraftMode.none,
        '1': PipelineDraftMode.normal,
        '2': PipelineDraftMode.custom
    }

    @staticmethod
    def convert_pipeline_type_to_str(pipeline_type_value):
        if pipeline_type_value in PipelineUtilities._pipeline_type_dict:
            return PipelineUtilities._pipeline_type_dict[pipeline_type_value].value
        elif isinstance(pipeline_type_value, enum.Enum):
            return pipeline_type_value.value
        else:
            return pipeline_type_value

    @staticmethod
    def convert_pipeline_draft_mode_to_str(pipeline_draft_mode_value):
        if pipeline_draft_mode_value in PipelineUtilities._pipeline_draft_mode_dict:
            return PipelineUtilities._pipeline_draft_mode_dict[pipeline_draft_mode_value].value
        elif isinstance(pipeline_draft_mode_value, enum.Enum):
            return pipeline_draft_mode_value.value
        else:
            return pipeline_draft_mode_value
