# Using LLMs in LIT on Google Cloud Platform

Architectural Notes

*   The `LitApp` HTTP API assumes that inputs will be passed around as
    identifiers and the reconsituted on the LitApp server before being sent to
    the model. The `model_server.py` will not have direct access to the loaded Datasets, and thus the HTTP API assumes that the JSON data passed to its endpoints will be the complete, reconstituted examples from the `LitApp`. The `model_server.py` will send back predictions in full JSON format.