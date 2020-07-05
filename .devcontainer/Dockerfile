FROM debian:9   

# sdk version
ARG SDK_VERSION_SHORT=15978945
ARG SDK_SOURCE=https://azuremlsdktestpypi.azureedge.net/Pipeline-Wrapper-SDK-Preview/$SDK_VERSION_SHORT
ARG SDK_VERSION_LONG=0.1.0.$SDK_VERSION_SHORT
ARG AZ_EXTENSION_SOURCE=https://azuremlsdktestpypi.azureedge.net/Pipeline-Wrapper-SDK-Preview/$SDK_VERSION_SHORT/azure_cli_ml-0.1.0.$SDK_VERSION_SHORT-py3-none-any.whl

# create vsonline
ARG USERNAME=vsonline
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid 800 docker && \
    apt-get update --fix-missing && \
    apt-get install -y apt-utils sudo apt-transport-https wget bzip2 ca-certificates curl gnupg-agent software-properties-common git vim htop tmux && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add - && \
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable" && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev && \
    cd ~ && \
    curl -O https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz && \
    tar -xf Python-3.7.3.tar.xz && \
    cd Python-3.7.3 && \
    ./configure && \
    make -j && \
    make install && \
    ln -s /usr/local/bin/python3.7 /usr/local/bin/python && \
    cd ~ && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    /usr/local/bin/python3.7 get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# create vsonline
RUN groupadd --gid $USER_GID $USERNAME && \ 
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    usermod -aG docker $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME
    
USER $USERNAME  

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /home/$USERNAME/.local/bin:$PATH

RUN /usr/local/bin/pip install --extra-index-url=$SDK_SOURCE azure-cli azureml-defaults==$SDK_VERSION_LONG azureml-pipeline-wrapper[notebooks]==$SDK_VERSION_LONG azureml-pipeline-core==$SDK_VERSION_LONG azure-storage-blob && \
    az extension add --source $AZ_EXTENSION_SOURCE --pip-extra-index-urls $SDK_SOURCE --yes && \
    whoami && \
    whoami 
RUN echo "PATH=$PATH" >> /home/$USERNAME/.bashrc

USER root
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
    

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]