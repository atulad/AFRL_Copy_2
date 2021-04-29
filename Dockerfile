# docker build: docker build -t afrl .
# docker run: nvidia-docker run --ipc=host -it --rm -v $(pwd):/code afrl:latest

ARG PARENT_IMAGE="nvcr.io/nvidia/pytorch:20.12-py3"
FROM $PARENT_IMAGE
# ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.6

ENV DEBIAN_FRONTEND="noninteractive" TZ="America/Houston"

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libglib2.0-0 && \
     rm -rf /var/lib/apt/lists/*

# Install Anaconda and dependencies
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -u -b -p  /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda install -y pytorch cudatoolkit=11.1 -c pytorch-nightly -c nvidia && \
    #  /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include && \
    #  /opt/conda/bin/conda install -y pytorch -c pytorch && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

ENV CODE_DIR /root/code

# Copy setup file only to install dependencies
# COPY ./setup.py ${CODE_DIR}/stable-baselines3/setup.py

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t robbyrussell && \
    echo "exec zsh" >> ~/.bashrc && \
    conda init zsh

RUN rm -r /workspace

CMD /bin/bash
