# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from traitlets import Unicode

from ipywidgets import DOMWidget, register


@register
class ValidateView(DOMWidget):
    """ A widget for the pipeline validate visualization.

    Especially, the widget accept message to update running status.
    And return the latest status into view.
    """

    _view_name = Unicode('ValidateView').tag(sync=True)
    _view_module = Unicode('validate_widget').tag(sync=True)
    _view_module_version = Unicode('0.0.0').tag(sync=True)

    graph_json = Unicode().tag(sync=True)
    env_json = Unicode().tag(sync=True)
    lib_url = Unicode().tag(sync=True)
    container_id = Unicode().tag(sync=True)
    visualize_id = Unicode().tag(sync=True)

    def __init__(self, **kwargs):
        """Create a ValidateView widget."""
        super(DOMWidget, self).__init__(**kwargs)
        self.graph_json = kwargs.get("graph_json")
        self.env_json = kwargs.get("env_json")
        self.lib_url = kwargs.get("lib_url")
        self.container_id = kwargs.get("container_id")
        self.visualize_id = kwargs.get("visualize_id")

        def handle_msg(instance, message, params):
            if message.get("message") == "log_query:request":
                self.handle_log_query_request(message.get("body"))

        self.on_msg(handle_msg)

    def handle_log_query_request(self, body):
        import requests
        uid = body.get("uid")
        content = body.get("content")
        response = requests.get(content.get("url"))
        self.send_message("log_query:response", {
            "uid": uid,
            "result": response.text
        })

    def send_message(self, message: str, content: dict):
        self.send({
            "message": message,
            "body": content
        })
