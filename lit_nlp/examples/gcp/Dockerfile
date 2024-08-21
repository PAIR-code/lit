# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Use the official lightweight Python image.
# https://hub.docker.com/_/python

FROM nvidia/cuda:12.5.1-base-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

ARG PYTHON_VERSION=python3.11

RUN apt-get update

# Install the CUDA Keyring package
# See https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu
RUN apt-get install -y curl gnupg ca-certificates
RUN curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -o cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

# Install system and CUDA packages
RUN apt-get install -y --no-install-recommends \
    cuda-command-line-tools-12-3 \
    cuda-cudart-dev-12-3 \
    cuda-nvcc-12-3 \
    cuda-cupti-12-3 \
    cuda-nvprune-12-3 \
    cuda-libraries-12-3 \
    cuda-nvrtc-12-3 \
    libcufft-12-3 \
    libcurand-12-3 \
    libcusolver-12-3 \
    libcusparse-12-3 \
    libcublas-12-3 \
    libcudnn8=8.9.6.50-1+cuda12.2 \
    libnvinfer-plugin8=8.6.1.6-1+cuda12.0 \
    libnvinfer8=8.6.1.6-1+cuda12.0 \
    build-essential \
    pkg-config \
    software-properties-common \
    unzip

# Install Python 3.11
RUN apt-get install -y --no-install-recommends \
    $PYTHON_VERSION \
    $PYTHON_VERSION-venv \
    $PYTHON_VERSION-distutils \
    $PYTHON_VERSION-dev
RUN ln -sf /usr/bin/$PYTHON_VERSION /usr/bin/python3
RUN ln -sf /usr/bin/$PYTHON_VERSION /usr/bin/python

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*



# ---- LIT on GCP from source ----

FROM base AS lit-gcp-dev
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install Node.js v18 (the base image ships with Node.js v12)
# See https://github.com/nodesource/distributions
RUN curl -fsSL https://deb.nodesource.com/setup_18.x -o nodesource_setup.sh
RUN bash nodesource_setup.sh
RUN apt-get install -y nodejs

# Install yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | \
    tee /etc/apt/sources.list.d/yarn.list
RUN apt update && apt -y install yarn

# TODO(b/353980272): Replace the default config with the GCP-specific config
COPY ./lit_nlp/examples/gunicorn_config.py ./

# TODO(b/353980272): Replace this with a requirements file specific to the GCP
# exmaple, this should include the core lit-nlp package.
COPY requirements_core.txt ./
COPY lit_nlp/examples/gcp/requirements.txt lit_nlp/examples/gcp/requirements.txt
RUN python -m pip install -r lit_nlp/examples/gcp/requirements.txt

# Copy the rest of the lit_nlp package
COPY . ./

# Build front-end with yarn
WORKDIR $APP_HOME/lit_nlp/client
ENV NODE_OPTIONS "--openssl-legacy-provider"
RUN yarn && yarn build && rm -rf node_modules/*

# TODO(b/353980272): Replace this with the GCP-specific config
# See https://github.com/PAIR-code/lit/blob/main/Dockerfile
WORKDIR $APP_HOME
ENTRYPOINT ["gunicorn", "--config=gunicorn_config.py"]
