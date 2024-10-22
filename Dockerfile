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

# ---- LIT Base Container ----

FROM python:3.11-slim AS lit-nlp-base

# Update Ubuntu packages and install basic utils
RUN apt-get update
RUN apt-get install -y wget curl gnupg2 gcc g++ git

# Copy local code to the container image.
ENV APP_HOME=/app
WORKDIR $APP_HOME

COPY ./lit_nlp/examples/gunicorn_config.py ./



# ---- LIT Container for Hosted Demos ----

FROM lit-nlp-base AS lit-nlp-prod

RUN python -m pip install 'lit-nlp[examples-discriminative-ai]'

WORKDIR $APP_HOME
ENTRYPOINT ["gunicorn", "--config=gunicorn_config.py"]



# ---- LIT Container for Developing and Testing Hosted Demos ----

FROM lit-nlp-base AS lit-nlp-dev

# Install yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | \
    tee /etc/apt/sources.list.d/yarn.list
RUN apt update && apt -y install yarn

# Set up python environment with production dependencies
# This step is slow as it installs many packages.
COPY requirements.txt \
     requirements_examples_common.txt \
     requirements_examples_discriminative_ai.txt \
     ./
RUN python -m pip install -r requirements_examples_discriminative_ai.txt

# Copy the rest of the lit_nlp package
COPY . ./

# Build front-end with yarn
WORKDIR $APP_HOME/lit_nlp/client
ENV NODE_OPTIONS="--openssl-legacy-provider"
RUN yarn && yarn build && rm -rf node_modules/*

# Run LIT server
# Note that the config file supports configuring the LIT demo that is launched
# via the DEMO_NAME and DEMO_PORT environment variables.
WORKDIR $APP_HOME
ENTRYPOINT ["gunicorn", "--config=gunicorn_config.py"]
