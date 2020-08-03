from .service_caller import DesignerServiceCaller


class _DesignerServiceCallerFactory:

    caller_cache_by_workspace_id = {}

    @staticmethod
    def get_instance(workspace):
        workspace_id = workspace._workspace_id
        cache = _DesignerServiceCallerFactory.caller_cache_by_workspace_id
        if workspace_id not in cache:
            cache[workspace_id] = DesignerServiceCaller(workspace)
        return cache[workspace_id]
