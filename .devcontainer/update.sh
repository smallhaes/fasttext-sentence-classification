#/bin/bash
USERNAME=vsonline
SDK_VERSION_SHORT=`curl -s https://versionofsdk.blob.core.windows.net/versionofsdk/version.txt`
SDK_SOURCE=https://azuremlsdktestpypi.azureedge.net/CLI-SDK-Runners-Validation/$SDK_VERSION_SHORT
SDK_VERSION_LONG=0.1.0.$SDK_VERSION_SHORT
AZ_EXTENSION_SOURCE=https://azuremlsdktestpypi.azureedge.net/CLI-SDK-Runners-Validation/$SDK_VERSION_SHORT/azure_cli_ml-0.1.0.$SDK_VERSION_SHORT-py3-none-any.whl

/home/$USERNAME/conda/bin/pip install -U --extra-index-url=$SDK_SOURCE azureml-defaults==$SDK_VERSION_LONG azureml-pipeline-wrapper[notebooks]==$SDK_VERSION_LONG azureml-pipeline-core==$SDK_VERSION_LONG
az extension remove -n azure-cli-ml 
az extension add --source $AZ_EXTENSION_SOURCE --pip-extra-index-urls $SDK_SOURCE --yes --debug