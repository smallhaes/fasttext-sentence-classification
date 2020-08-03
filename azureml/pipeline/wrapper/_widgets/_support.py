# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from traitlets import Unicode
from ipywidgets import DOMWidget, register


@register
class SupportDetectView(DOMWidget):
    """ A widget for detect widget support.

    We need to detect the current environment support widget or not.
    This widget will send support message after loaded and rendered to tell us it work.
    """

    _view_name = Unicode('SupportDetectView').tag(sync=True)
    _view_module = Unicode('support_detect_widget').tag(sync=True)

    callback = None

    def __init__(self, **kwargs):
        """Create a SupportDetectView widget."""
        super(DOMWidget, self).__init__(**kwargs)
        self.callback = kwargs.get("callback")

        def handle_msg(instance, message, params):
            self.callback()

        self.on_msg(handle_msg)
