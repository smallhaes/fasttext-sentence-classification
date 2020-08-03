# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
from datetime import datetime
import concurrent.futures

from ._module import _OutputBuilder, _InputBuilder
from ._module_run_helper import _module_run, _prepare_module_inputs


NODE_ID = 'node_id'
STEP_PREFIX = 'prefix'
WORKING_DIR = 'working_dir'
RUN_ID = 'run_id'
EXECUTION_LOG = 'executionlogs.txt'

datetime_format = '%Y-%m-%d %H:%M:%S'
submit_log_format = '[{}] Submitting {} runs, first five are: {} \n'
complete_log_format = '[{}] Completing processing run id {}\n'
failed_log_format = '[{}] Execution of experiment failed, update experiment status and cancel running nodes.'


def _orchestrate_pipeline_run(pipeline, working_dir, run, module_node_to_graph_node_mapping, visualizer=None,
                              pipeline_parameters=None, show_output=False,
                              continue_on_step_failure=None, max_workers=None):
    """
    Orchestrate pipeline run

    Orchestrating pipeline run to make steps executing in parallel. Firstly will submit no dependency
    steps to start pipeline run, using threadpool to parallel execute steps. When previous steps completed,
    will push no dependency steps to threadpool, until all steps completed.

    :param pipeline: Orchestrated pipeline
    :type pipeline: azureml.pipeline.wrapper.Pipeline
    :param working_dir: pipline run data and snapshot store path
    :type working_dir: str
    :param module_node_to_graph_node_mapping: mapping of module node to graph node
    :type module_node_to_graph_node_mapping: dict
    :param visualizer: To show pipeline graph in notebook
    :type visualizer: azureml.pipeline.wrapper._widgets._visualize
    :param pipeline_parameters: An optional dictionary of pipeline parameter
    :type pipeline_parameters: dict({str:str})
    :param show_output: Indicates whether to show the pipeline run status on sys.stdout.
    :type show_output: bool
    :param continue_on_step_failure: Indicates whether to continue pipeline execution if a step fails.
        If True, only steps that have no dependency on the output of the failed step will continue execution.
    :type continue_on_step_failure: bool
    :param max_workers:  The maximum number of threads that can be used to execute pipeline steps.
        If max_workers is None, it will default to the number of processors on the machine.
    :type max_workers: int
    :return: whether pipeline run successful finished
    :rtype: bool
    """
    # prepare for node run
    node_list, module_to_node_mapping = pipeline._expand_pipeline_nodes('', module_node_to_graph_node_mapping)
    node_dict = {node._instance_id: node for node in node_list}
    node_output_dict, begin_exec_node = _prepare_pipeline_run(node_dict)
    executed_nodes = []
    pipeline_run_success = True

    # download node input datset
    print('Preparing pipeline dataset.')
    for node in node_list:
        for input_name, input_value in node.inputs.items():
            _prepare_module_inputs(
                pipeline.workspace, input_name, input_value.dset, working_dir,
                pipeline_parameters, module_to_node_mapping)
    print('Prepared pipeline dataset.')

    # start running node
    execution_log_path = os.path.join(working_dir, EXECUTION_LOG)
    with open(execution_log_path, 'w') as execution_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for node in begin_exec_node:
                child_run = run.child_run(name=node_dict[node].name)
                submit_future = executor.submit(
                    exec_node, node_dict[node], child_run, working_dir, pipeline_parameters,
                    module_to_node_mapping, show_output, visualizer)
                futures[submit_future] = {
                    NODE_ID: node,
                    RUN_ID: child_run.id
                }
            if len(begin_exec_node) > 0:
                execution_file.write(
                    submit_log_format.format(datetime.now().strftime(datetime_format),
                                             len(begin_exec_node),
                                             ','.join([value[RUN_ID] for value in futures.values()][0:5])))

            current_futures = futures.keys()
            while current_futures:
                done_futures, current_futures = concurrent.futures.wait(
                    current_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                # update running node task list
                next_node_list = set()
                for future in done_futures:
                    if future.result() != 'Completed':
                        pipeline_run_success = False
                        execution_file.write(failed_log_format.format(datetime.now().strftime(datetime_format)))
                        continue
                    node_id = futures[future][NODE_ID]
                    execution_file.write(
                        complete_log_format.format(datetime.now().strftime(datetime_format),
                                                   futures[future][RUN_ID]
                                                   ))
                    executed_nodes.append(node_id)
                    if node_id in node_output_dict:
                        next_node_list.update(node_output_dict[node_id])
                if not pipeline_run_success and not continue_on_step_failure:
                    concurrent.futures.wait(current_futures, return_when=concurrent.futures.ALL_COMPLETED)
                    break
                else:
                    next_nodes = _find_next_run_node(next_node_list, executed_nodes, node_dict)
                    next_futures = {}
                    for node in next_nodes:
                        child_run = run.child_run(name=node_dict[node].name)
                        future = executor.submit(
                            exec_node, node_dict[node], child_run, working_dir, pipeline_parameters,
                            module_to_node_mapping, show_output, visualizer)
                        next_futures[future] = {NODE_ID: node, RUN_ID: child_run.id}
                        current_futures.add(future)
                    if len(next_nodes) > 0:
                        execution_file.write(
                            submit_log_format.format(datetime.now().strftime(datetime_format),
                                                     len(next_nodes),
                                                     ','.join([value[RUN_ID] for value in next_futures.values()][0:5])
                                                     ))
                    futures.update(next_futures)
    run.upload_file(EXECUTION_LOG, execution_log_path)
    return pipeline_run_success


def get_node_input_dset(input_dset):
    if isinstance(input_dset, _InputBuilder):
        return get_node_input_dset(input_dset.dset)
    else:
        return input_dset


def _prepare_pipeline_run(node_dict):
    node_output_dict = {}
    begin_exec_node = []
    for node in node_dict.values():
        pre_input_list = []
        for input in node.inputs.values():
            dset = get_node_input_dset(input.dset)
            if isinstance(dset, _OutputBuilder):
                pre_input_list.append(dset)
        if len(pre_input_list) == 0:
            begin_exec_node.append(node._instance_id)
        for input in pre_input_list:
            if input.module_instance_id not in node_output_dict.keys():
                node_output_dict[input.module_instance_id] = []
            node_output_dict[input.module_instance_id].append(node._instance_id)

    return node_output_dict, begin_exec_node


def _find_next_run_node(next_node_list, executed_nodes, node_dict):
    next_nodes = set()
    for node_id in next_node_list:
        node = node_dict[node_id]
        node_inputs = [get_node_input_dset(input) for input in node.inputs.values()]
        if all([input.module_instance_id in executed_nodes
                for input in node_inputs if isinstance(input, _OutputBuilder)]):
            next_nodes.add(node._instance_id)
    return next_nodes


def exec_node(node, run, working_dir, pipeline_parameters, module_to_node_mapping, show_output, visualizer):
    try:
        node_working_dir = os.path.join(working_dir,
                                        module_to_node_mapping[node._instance_id][STEP_PREFIX],
                                        f'{node.name}_{run.id}')
        use_docker = True
        if '_TEST_ENV' in os.environ:
            use_docker = False
        status = _module_run(
            module=node,
            working_dir=node_working_dir,
            run=run,
            use_docker=use_docker,
            node_id=module_to_node_mapping[node._instance_id][NODE_ID],
            visualizer=visualizer,
            show_output=show_output,
            module_to_node_mapping=module_to_node_mapping,
            data_dir=working_dir,
            pipeline_parameters=pipeline_parameters)
        if status != 'Completed':
            raise RuntimeError(f'Step f"{node.name}_{run.id}" run failed.')
        module_to_node_mapping[node._instance_id][WORKING_DIR] = node_working_dir
    except Exception as e:
        print(e)
        print(f'{node.name} run failed, exception: {str(e)}')
        return 'Failed'
    return status
