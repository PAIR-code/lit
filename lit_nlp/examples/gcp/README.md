# Using LLMs in LIT on Google Cloud Platform

## Developing

### Use a virtual environment

```shell
# Create and activate the virtual environment
python3 -m venv ~/.venvs/lit-on-gcp
source ~/.venvs/lit-on-gcp/bin/activate

# Install the requirements and LIT in editable mode
pip install -f ./lit_nlp/examples/gcp/requirements.txt
pip install -e .

# Optionally, install tetsing requirements
pip install -f ./requirements_test.txt
```

### Build the Docker image

```shell
docker build -f ./lit_nlp/examples/gcp/Dockerfile -t lit-app:gcp-dev .
```

### Run GPT-2 in a Docker container

```shell
# Runs GPT-2 in Keras on Tensorflow
docker run --rm -p 5432:5432 -e MODEL_CONFIG=gpt2:gpt2_base_en lit-app:gcp-dev
```

Architectural Notes

*   The `LitApp` HTTP API assumes that inputs will be passed around as
    identifiers and the reconsituted on the LitApp server before being sent to
    the model. The `model_server.py` will not have direct access to the loaded Datasets, and thus the HTTP API assumes that the JSON data passed to its endpoints will be the complete, reconstituted examples from the `LitApp`. The `model_server.py` will send back predictions in full JSON format.