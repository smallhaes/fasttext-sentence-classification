# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import List


class MessageProxy:
    def send_message(self, message: str, content: dict):
        raise NotImplementedError("send_message is abstract")


class _WidgetMessageProxy(MessageProxy):
    def __init__(self, instance):
        self._instance = instance

    def send_message(self, message: str, content: dict):
        self._instance.send_message(message, content)


class _ScriptMessageProxy(MessageProxy):
    def __init__(self, container_id):
        self._container_id = container_id

    def send_message(self, message: str, content: dict):
        # from IPython.display import Javascript, display
        # message_script = Javascript(
        #     '''
        #     (function () {{
        #         if (!window.__event_hub) {{
        #             window.__event_hub = {{}}
        #         }}
        #         if (!window.__event_hub[{0}]) {{
        #             window.__event_hub[{0}] = {{}}
        #         }}

        #         var message = {1}
        #         var body = {2}

        #         if (!window.__event_hub[{0}][message]) {{
        #             window.__event_hub[{0}][message] = []
        #         }}
        #         var listeners = window.__event_hub[{0}][message]

        #         listeners.forEach(cb => {{
        #             try {{
        #                 cb(body)
        #             }} catch (e) {{
        #                 console.error("Unexpected error in listener", e)
        #             }}
        #         }})

        #         console.log(body)
        #     }})()
        #     '''.format(json.dumps(self._container_id), json.dumps(message), json.dumps(content))
        # )

        # display(message_script)
        pass


class _RunStatusVisualizer:
    def __init__(self, proxies: List[MessageProxy]):
        self._proxies = proxies

    def send_message(self, message: str, content: dict):
        for proxy in self._proxies:
            proxy.send_message(message, content)
