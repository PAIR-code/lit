# Copyright 2023 Google LLC
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
FROM python:3.10-slim

# Update Ubuntu packages and install basic utils
RUN apt-get update
RUN apt-get install -y wget curl gnupg2 gcc g++ git

# Install yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | \
    tee /etc/apt/sources.list.d/yarn.list
RUN apt update && apt -y install yarn

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

# Set up python environment with production dependencies
# This step is slow as it installs many packages.
COPY ./requirements*.txt ./
RUN python -m pip install -r requirements.txt

# Build front-end with yarn
COPY . ./
WORKDIR /app/lit_nlp/client
ENV NODE_OPTIONS "--openssl-legacy-provider"
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
