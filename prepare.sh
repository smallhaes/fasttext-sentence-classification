echo 'pythonenv3.7' >> .gitignore
echo 'get-pip.py' >> .gitignore
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
python3 -m pip install  --extra-index-url=https://azuremlsdktestpypi.azureedge.net/CLI-SDK-Runners-Validation/15886596 -r requirements.txt