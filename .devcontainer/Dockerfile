FROM debian:9   

# user information
ARG USERNAME=vsonline
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# miniconda
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh
ARG CONDA_DIR=/home/$USERNAME/.conda
ARG CONDA_ENV_NAME=modulesdkpreview
ARG CONDA_ENV_DIR=$CONDA_DIR/envs/$CONDA_ENV_NAME
ARG PYTHON_VERSION=3.7

# sdk version
ARG SDK_VERSION_URL=https://modulesdkpreview.blob.core.windows.net/sdk/preview/version.txt

# env
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH $CONDA_DIR/bin:$CONDA_ENV_DIR/bin:$PATH

# install common softwares
RUN groupadd --gid 800 docker && \
    apt-get update --fix-missing && \
    apt-get install -y apt-utils sudo apt-transport-https wget bzip2 ca-certificates curl gnupg-agent software-properties-common git vim htop tmux zsh

# init
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
