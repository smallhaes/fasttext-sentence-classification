# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import os
import time
from uuid import uuid4
from .._loggerfactory import _LoggerFactory, track
from .._utils import _in_jupyter_nb

_logger = None


def _get_logger():
    global _logger
    if _logger is not None:
        return _logger
    _logger = _LoggerFactory.get_logger(__name__)
    return _logger


def _visualize(graphyaml: dict, envinfo: dict = None, is_prod: bool = False):
    if not _in_jupyter_nb():
        return

    from IPython.display import Javascript, display

    visualize_id = str(uuid4())
    container_id = "container_id_{0}".format(visualize_id)
    widget_container_id = "{0}_widget".format(container_id)
    script_container_id = "{0}_script".format(container_id)
    # Todo: This blob storage is for demo only. Use real cdn endpoint for production.
    lib_url = _get_ux_lib_url(is_prod)

    js_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js/validation.js")
    widget_view_lib = Javascript(filename=js_filename)
    display(widget_view_lib)

    graphjson = json.dumps(graphyaml)

    envjson = json.dumps(envinfo) if envinfo is not None else '{}'

    display_widget = display_graph_by_widget(graphjson=graphjson, lib_url=lib_url, visualize_id=visualize_id,
                                             container_id=widget_container_id, envjson=envjson)
    display(display_widget)
    time.sleep(1)

    insert_html = display_graph_by_insert(graphjson=graphjson, lib_url=lib_url, visualize_id=visualize_id,
                                          container_id=script_container_id, envjson=envjson)
    display(insert_html)

    from ._run_status import _RunStatusVisualizer, _WidgetMessageProxy, _ScriptMessageProxy
    visualizer = _RunStatusVisualizer([_WidgetMessageProxy(instance=display_widget),
                                       _ScriptMessageProxy(container_id=container_id)])
    return visualizer


def display_graph_by_widget(graphjson, lib_url, container_id, envjson, visualize_id):
    from ._validation import ValidateView
    return ValidateView(graph_json=graphjson, env_json=envjson, lib_url=lib_url, container_id=container_id,
                        visualize_id=visualize_id)


def display_graph_by_insert(graphjson, lib_url, container_id, envjson, visualize_id):
    from IPython.display import HTML
    return HTML(
        '''
        <style>
        #{2} svg.react-dag-editor-svg-container {{
            height: 800px;
        }}
        </style>
        <div id="{2}"></div>
        <script>
            (function () {{
                if (!window._renderLock) {{
                    window._renderLock = {{}}
                }}
                if (window._renderLock[{5}]) {{
                    return
                }}
                window._renderLock[{5}] = "script"
                console.log("load as script", Date.now())

                window.render_container_id={3};
                window.graph_json={0};
                window.env_json={4};
                window.before_script = performance.now();

                var script = document.createElement('script')
                script.src = "{1}"
                document.getElementById({3}).appendChild(script)
            }})()
        </script>
        '''.format(graphjson, lib_url, container_id, json.dumps(container_id), envjson, json.dumps(visualize_id))
    )


def _get_ux_lib_url(is_prod: bool):
    return _get_ux_prod_lib_url() if is_prod else _get_ux_test_lib_url()


def _get_ux_test_lib_url():
    return 'https://yucongj-test.azureedge.net/libs/test/index.js?t={}'.format(int(time.time()))


@track(_get_logger)
def _get_ux_prod_lib_url():
    try:
        from packaging.version import parse
        from packaging.specifiers import SpecifierSet
    except ImportError as e:
        print("Couldn't import {0}. Please ensure {0} is installed."
              "Try install azureml-pipeline-wrapper[notebooks].".format(e.name))
        raise e

    # update specifier every time release
    specifier_set = SpecifierSet("~=0.0.0")

    account_name = "yucongjteststorage"
    prod_prefix = "prod"
    container_name = "libs"
    blobs_list = try_get_blobs_list(account_name=account_name,
                                    container_name=container_name,
                                    prod_prefix=prod_prefix)

    result_version = None
    for blob in blobs_list:
        path_list = _split_all(blob)

        if len(path_list) >= 2 and path_list[0] == prod_prefix:
            version_str = path_list[1]
            version = parse(version_str)

            result_version = version if version in specifier_set and \
                (result_version is None or version > result_version) else result_version

    if result_version is None:
        raise Exception("cannot find target version")

    result_version_str = str(result_version)
    _LoggerFactory.trace(_get_logger(), "Pipeline_fetch_ux_production_bundle", {
        'result_version': result_version_str
    })

    return 'https://yucongj.azureedge.net/libs/prod/{0}/index.js'.format(result_version_str)


def try_get_blobs_list(account_name: str, container_name: str, prod_prefix: str):
    try:
        try:
            return get_blobs_list_by_v12(account_name=account_name,
                                         container_name=container_name,
                                         prod_prefix=prod_prefix)
        except ImportError:
            return get_blobs_list_by_v2(account_name=account_name,
                                        container_name=container_name,
                                        prod_prefix=prod_prefix)
    except ImportError as e:
        print("Couldn't import expected version of 'azure-storage-blob'."
              "Please ensure 'azure-storage-blob' is installed."
              "Try install azureml-pipeline-wrapper[notebooks].".format(e.name))
        raise e


def get_blobs_list_by_v12(account_name: str, container_name: str, prod_prefix: str):
    from azure.storage.blob import ContainerClient

    container_url = "https://{}.blob.core.windows.net/{}".format(account_name, container_name)
    container_client = ContainerClient.from_container_url(container_url)
    blobs_list = container_client.list_blobs(name_starts_with=prod_prefix)
    return [blob.name for blob in blobs_list]


def get_blobs_list_by_v2(account_name: str, container_name: str, prod_prefix: str):
    from azure.storage.blob import BlockBlobService

    block_blob_service = BlockBlobService(account_name=account_name)
    blobs_list = list(block_blob_service.list_blob_names(container_name=container_name, prefix=prod_prefix))
    return blobs_list


# copy from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
def _split_all(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts
