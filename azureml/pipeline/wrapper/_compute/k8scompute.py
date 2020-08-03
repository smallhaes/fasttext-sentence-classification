# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing Azure Machine Learning compute targets in Azure Machine Learning."""

import json
import requests
from azureml._compute._constants import MLC_COMPUTE_RESOURCE_ID_FMT
from azureml._compute._constants import MLC_WORKSPACE_API_VERSION
from azureml.core.compute import ComputeTarget
from azureml.exceptions import ComputeTargetException


class AksCompute(ComputeTarget):
    """AksCompute is a customer managed Kubernetes infrastructure, which can be created and attached to workspace
    by cluster admin. User granted access and quota to the compute can easily specify and submit a one-node or
    distributed multi-node training job to the compute. The compute executes in a containerized environment and
    packages your model dependencies in a docker container.
    For more information, see `What are compute targets in Azure Machine
    Learning? <https://docs.microsoft.com/azure/machine-learning/concept-compute-target>`_

    .. remarks::

        In the following example, a persistent compute target provisioned by
        :class:`azureml.contrib.core.compute.k8scompute.AksCompute` is created.

    :param workspace: The workspace object containing the AksCompute object to retrieve.
    :type workspace: azureml.core.Workspace
    :param name: The name of the of the AksCompute object to retrieve.
    :type name: str
    """

    _compute_type = 'Cmk8s'

    def _initialize(self, workspace, obj_dict):
        """Initialize implementation method.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        name = obj_dict['name']
        compute_resource_id = MLC_COMPUTE_RESOURCE_ID_FMT.format(workspace.subscription_id, workspace.resource_group,
                                                                 workspace.name, name)
        resource_manager_endpoint = self._get_resource_manager_endpoint(workspace)
        mlc_endpoint = '{}{}'.format(resource_manager_endpoint, compute_resource_id)
        location = obj_dict['location']
        compute_type = obj_dict['properties']['computeType']
        tags = obj_dict['tags']
        description = obj_dict['properties']['description']
        created_on = obj_dict['properties'].get('createdOn')
        modified_on = obj_dict['properties'].get('modifiedOn')
        cluster_resource_id = obj_dict['properties']['resourceId']
        cluster_location = obj_dict['properties']['computeLocation'] \
            if 'computeLocation' in obj_dict['properties'] else None
        provisioning_state = obj_dict['properties']['provisioningState']
        provisioning_errors = obj_dict['properties']['provisioningErrors']
        is_attached = obj_dict['properties']['isAttachedCompute'] \
            if 'isAttachedCompute' in obj_dict['properties'] else None
        connection_string = obj_dict['properties']['properties']['connectionString'] \
            if obj_dict['properties']['properties'] else None
        super(AksCompute, self)._initialize(compute_resource_id, name, location, compute_type, tags, description,
                                            created_on, modified_on, provisioning_state, provisioning_errors,
                                            cluster_resource_id, cluster_location, workspace, mlc_endpoint, None,
                                            workspace._auth, is_attached)

        self.connection_string = connection_string

    def __repr__(self):
        """Return the string representation of the AksCompute object.

        :return: String representation of the AksCompute object
        :rtype: str
        """
        return super().__repr__()

    @staticmethod
    def attach(workspace, name, attach_configuration):
        """Associate an existing AzureML Kubernetes compute resource with the provided workspace.

        :param workspace: The workspace object to associate the compute resource with.
        :type workspace: azureml.core.Workspace
        :param name: The name to associate with the compute resource inside the provided workspace. Does not have to
            match the name of the compute resource to be attached..
        :type name: str
        :param attach_configuration: Attach configuration object.
        :type attach_configuration: azureml.contrib.core.compute.k8scompute.AksComputeAttachConfiguration
        :return: An AksCompute object representation of the compute object.
        :rtype: azureml.contrib.core.compute.k8scompute.AksCompute
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        raise ComputeTargetException('Attach is not supported for AksCompute object.')

    def refresh_state(self):
        """Perform an in-place update of the properties of the object.

        This method updates the properties based on the current state of the corresponding cloud object.
        This is primarily used for manual polling of compute state.
        """
        cluster = AksCompute(self.workspace, self.name)
        self.modified_on = cluster.modified_on
        self.provisioning_state = cluster.provisioning_state
        self.provisioning_errors = cluster.provisioning_errors
        self.cluster_resource_id = cluster.cluster_resource_id
        self.cluster_location = cluster.cluster_location
        self.connection_string = cluster.connection_string

    def delete(self):
        """Delete is not supported for an AksCompute object. Use :meth:`detach` instead.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        raise ComputeTargetException('Delete is not supported for AksCompute object.')

    def detach(self):
        """Detach the AksCompute object from its associated workspace.

        Underlying cloud objects are not deleted, only the association is removed.

        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        raise ComputeTargetException('Detach is not supported for AksCompute object.')

    def get_credentials(self):
        """Retrieve the credentials for the AzureML Kubernetes target.

        :return: Credentials for the AzureML Kubernetes target
        :rtype: dict
        :raises: ComputeTargetException
        """
        endpoint = self._mlc_endpoint + '/listKeys'
        headers = self._auth.get_authentication_header()
        params = {'api-version': MLC_WORKSPACE_API_VERSION}
        resp = requests.post(endpoint, params=params, headers=headers)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ComputeTargetException('Received bad response from MLC:\n'
                                         'Response Code: {}\n'
                                         'Headers: {}\n'
                                         'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        creds_content = json.loads(content)
        return creds_content

    def serialize(self):
        """Convert this AksCompute object into a JSON serialized dictionary.

        :return: The JSON representation of this AksCompute object.
        :rtype: dict
        """
        AksCompute_properties = {'connectionString': self.connection_string}
        cluster_properties = {'computeType': self.type, 'computeLocation': self.cluster_location,
                              'description': self.description, 'resourceId': self.cluster_resource_id,
                              'provisioningErrors': self.provisioning_errors,
                              'provisioningState': self.provisioning_state, 'properties': AksCompute_properties}
        return {'id': self.id, 'name': self.name, 'tags': self.tags, 'location': self.location,
                'properties': cluster_properties}

    @staticmethod
    def deserialize(workspace, object_dict):
        """Convert a JSON object into an AksCompute object.

        .. remarks::

            Raises a :class:`azureml.exceptions.ComputeTargetException` if the provided
            workspace is not the workspace the Compute is associated with.

        :param workspace: The workspace object the AksCompute object is associated with.
        :type workspace: azureml.core.Workspace
        :param object_dict: A JSON object to convert to a AksCompute object.
        :type object_dict: dict
        :return: The AksCompute representation of the provided JSON object.
        :rtype: azureml.contrib.core.compute.k8scompute.AksCompute
        :raises: :class:`azureml.exceptions.ComputeTargetException`
        """
        AksCompute._validate_get_payload(object_dict)
        target = AksCompute(None, None)
        target._initialize(workspace, object_dict)
        return target

    @staticmethod
    def _validate_get_payload(payload):
        if 'properties' not in payload or 'computeType' not in payload['properties']:
            raise ComputeTargetException('Invalid cluster payload:\n'
                                         '{}'.format(payload))
        if payload['properties']['computeType'] != AksCompute._compute_type:
            raise ComputeTargetException('Invalid cluster payload, not "{}":\n'
                                         '{}'.format(AksCompute._compute_type, payload))
        for arm_key in ['location', 'id', 'tags']:
            if arm_key not in payload:
                raise ComputeTargetException('Invalid cluster payload, missing ["{}"]:\n'
                                             '{}'.format(arm_key, payload))
        for key in ['properties', 'provisioningErrors', 'description', 'provisioningState', 'resourceId']:
            if key not in payload['properties']:
                raise ComputeTargetException('Invalid cluster payload, missing ["properties"]["{}"]:\n'
                                             '{}'.format(key, payload))
