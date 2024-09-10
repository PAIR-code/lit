"""Wrapper for connetecting to LLMs on GCP via the model_server HTTP API."""

import enum

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.api.types import Spec
from lit_nlp.examples.gcp import constants as lit_gcp_constants
from lit_nlp.examples.prompt_debugging import constants as pd_constants
from lit_nlp.examples.prompt_debugging import utils as pd_utils
from lit_nlp.lib import serialize
import requests

"""
Plan for this module:

From GitHub:

*   Rebase to include cl/672527408 and the CL described above
*   Define an enum to track HTTP endpoints across Python modules
*   Adopt HTTP endpoint enum across model_server.py and LlmOverHTTP
*   Adopt model_specs.py in LlmOverHTTP, using HTTP endpoint enum for
    conditional additions

"""

_LlmHTTPEndpoints = lit_gcp_constants.LlmHTTPEndpoints


class LlmOverHTTP(lit_model.BatchedRemoteModel):

  def __init__(
    self,
    base_url: str,
    endpoint: str | _LlmHTTPEndpoints,
    max_concurrent_requests: int = 4,
    max_qps: int | float = 25
  ):
    super().__init__(max_concurrent_requests, max_qps)
    self.endpoint = _LlmHTTPEndpoints(endpoint)
    self.url = f'{base_url}/{self.endpoint.value}'

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
    """
    response = requests.post(
        self.url, data=serialize.to_json(list(inputs),  simple=True)
    )

    if not (200 <= response.status_code < 300):
      raise RuntimeError()

    outputs = serialize.from_json(response.text)
    return outputs


def initialize_model_group_for_salience(
    name: str, base_url: str, *args, **kw
) -> dict[str, lit_model.Model]:
  """Creates '{name}' and '_{name}_salience' and '_{name}_tokenizer'."""
  salience_name, tokenizer_name = pd_utils.generate_model_group_names(name)

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
      name: generation_model,
      salience_name: salience_model,
      tokenizer_name: tokenizer_model,
  }
