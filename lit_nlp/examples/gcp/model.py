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

"""Wrapper for connecting to LLMs on GCP via the model_server HTTP API."""

from lit_nlp import app as lit_app
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.gcp import constants as lit_gcp_constants
from lit_nlp.examples.prompt_debugging import constants as pd_constants
from lit_nlp.examples.prompt_debugging import utils as pd_utils
from lit_nlp.lib import serialize
import requests

_LlmHTTPEndpoints = lit_gcp_constants.LlmHTTPEndpoints

LLM_ON_GCP_INIT_SPEC: lit_types.Spec = {
    # Note that `new_name` is not actually passed to LlmOverHTTP but the
    # `/create_model` API will validate the config with a `new_name` in it.
    'new_name': lit_types.String(required=False),
    'base_url': lit_types.String(),
    'identity_token': lit_types.String(default=''),
    'max_concurrent_requests': lit_types.Integer(default=1),
    'max_qps': lit_types.Integer(default=25, required=False),
}


class LlmOverHTTP(lit_model.BatchedRemoteModel):
  """Model wrapper LLMs hosted in a Model Server container."""

  def __init__(
      self,
      base_url: str,
      identity_token: str,
      endpoint: str | _LlmHTTPEndpoints,
      max_concurrent_requests: int = 4,
      max_qps: int | float = 25,
  ):
    super().__init__(max_concurrent_requests, max_qps)
    self.endpoint = _LlmHTTPEndpoints(endpoint)
    self.url = f'{base_url}/{self.endpoint.value}'
    self.identity_token = identity_token

  def input_spec(self) -> lit_types.Spec:
    input_spec = pd_constants.INPUT_SPEC

    if self.endpoint == _LlmHTTPEndpoints.SALIENCE:
      input_spec |= pd_constants.INPUT_SPEC_SALIENCE

    return input_spec

  def output_spec(self) -> lit_types.Spec:
    if self.endpoint == _LlmHTTPEndpoints.GENERATE:
      return (
          pd_constants.OUTPUT_SPEC_GENERATION
          | pd_constants.OUTPUT_SPEC_GENERATION_EMBEDDINGS
      )
    elif self.endpoint == _LlmHTTPEndpoints.SALIENCE:
      return pd_constants.OUTPUT_SPEC_SALIENCE
    else:
      return pd_constants.OUTPUT_SPEC_TOKENIZER

  def predict_minibatch(
      self, inputs: list[lit_types.JsonDict]
  ) -> list[lit_types.JsonDict]:
    """Run prediction on a batch of inputs.

    Subclass should implement this.

    Args:
      inputs: sequence of inputs, following model.input_spec()

    Returns:
      list of outputs, following model.output_spec()

    Raises:
      RuntimeError: Received non-200 HTTP Status Codes in the response.
    """
    inputs = {'inputs': inputs}
    headers = {
        'Authorization': f'Bearer {self.identity_token}',
        'Content-Type': 'application/json',
    }
    response = requests.post(
        self.url, headers=headers, data=serialize.to_json(inputs, simple=True)
    )

    if not (200 <= response.status_code < 300):
      raise RuntimeError()

    outputs = serialize.from_json(response.text)
    return outputs


def initialize_model_group_for_salience(
    new_name: str, base_url: str, *args, **kw
) -> lit_model.ModelMap:
  """Creates '{name}' and '_{name}_salience' and '_{name}_tokenizer'."""
  salience_name, tokenizer_name = pd_utils.generate_model_group_names(new_name)

  generation_model = LlmOverHTTP(
      *args, base_url=base_url, endpoint=_LlmHTTPEndpoints.GENERATE, **kw
  )
  salience_model = LlmOverHTTP(
      *args, base_url=base_url, endpoint=_LlmHTTPEndpoints.SALIENCE, **kw
  )
  tokenizer_model = LlmOverHTTP(
      *args, base_url=base_url, endpoint=_LlmHTTPEndpoints.TOKENIZE, **kw
  )

  return {
      new_name: generation_model,
      salience_name: salience_model,
      tokenizer_name: tokenizer_model,
  }


def get_model_loaders() -> lit_app.ModelLoadersMap:
  return {
      'LLM (self hosted)': (
          initialize_model_group_for_salience,
          LLM_ON_GCP_INIT_SPEC,
      )
  }
