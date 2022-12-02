# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Update Ubuntu packages and install basic utils
RUN apt-get update
RUN apt-get install -y wget curl gnupg2 gcc g++

# Install yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | \
    tee /etc/apt/sources.list.d/yarn.list
RUN apt update && apt -y install yarn

# Install Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3 \
    && rm Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH /opt/anaconda3/bin:$PATH

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Set up conda environment with production dependencies
# This step is slow as it installs many packages.
RUN conda env create -f environment.yml

# Workaround for 'conda activate' depending on shell features
# that don't necessarily work in Docker.
# This simulates the effect of 'conda activate'
# See https://github.com/ContinuumIO/docker-images/issues/89
# If this breaks in a future version of conda, add
#   RUN conda shell.posix activate lit-nlp
# to see what conda activate lit-nlp would do, and update the commands below
# accordingly.
ENV PATH /opt/anaconda3/envs/lit-nlp/bin:$PATH
ENV CONDA_PREFIX "/opt/anaconda3/envs/lit-nlp"
ENV CONDA_SHLVL "1"
ENV CONDA_DEFAULT_ENV "lit-nlp"

# Build front-end with yarn
WORKDIR lit_nlp/client
RUN yarn && yarn build && rm -rf node_modules/*
WORKDIR $APP_HOME

# Default demo app command to run.
ARG DEFAULT_DEMO="glue_demo"
ENV DEMO_NAME $DEFAULT_DEMO

ARG DEFAULT_PORT="5432"
ENV DEMO_PORT $DEFAULT_PORT

# Run LIT server
ENTRYPOINT exec gunicorn \
           -c lit_nlp/examples/gunicorn_config.py \
           --bind="0.0.0.0:$DEMO_PORT" \
           "lit_nlp.examples.$DEMO_NAME:get_wsgi_app()"
