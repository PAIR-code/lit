"""Model Wrapper for generative models."""

from collections.abc import Iterable
import logging
import time
from typing import Optional, Union
from vertexai import generative_models
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types

_MAX_NUM_RETRIES = 5

_DEFAULT_CANDIDATE_COUNT = 1

_DEFAULT_MAX_OUTPUT_TOKENS = 256


class VertexModelGardenModel(lit_model.BatchedRemoteModel):
  """VertexModelGardenModel is a wrapper for Vertex AI Model Garden model.

  Attributes:
    model_name: The name of the model to load.
    max_concurrent_requests: The maximum number of concurrent requests to the
      model.
    max_qps: The maximum number of queries per second to the model.
    temperature: The temperature to use for the model.
    candidate_count: The number of candidates to generate.
    max_output_tokens: The maximum number of tokens to generate.

  Please note the model will predict all examples at a fixed temperature.
  """

  def __init__(
      self,
      model_name: str,
      max_concurrent_requests: int = 4,
      max_qps: Union[int, float] = 25,
      temperature: Optional[float] = None,
      candidate_count: Optional[int] = _DEFAULT_CANDIDATE_COUNT,
      max_output_tokens: Optional[int] = _DEFAULT_MAX_OUTPUT_TOKENS,
  ):
    super().__init__(max_concurrent_requests, max_qps)
    # Connect to the remote model.
    self._generation_config = generative_models.GenerationConfig(
        temperature=temperature,
        candidate_count=candidate_count,
        max_output_tokens=max_output_tokens,
    )
    self._model = generative_models.GenerativeModel(model_name)

  # TODO(fanyeycourage): Enable query_model to take a list of input_text, and
  # return a list of predictions.
  def query_model(self, input_text: str) -> lit_types.ScoredTextCandidates:
    num_attempts = 0
    predictions = None
    exception = None

    while num_attempts < _MAX_NUM_RETRIES and predictions is None:
      num_attempts += 1

      try:
        predictions = self._model.generate_content(
            input_text,
            generation_config=self._generation_config,
        )
      except Exception as e:  # pylint: disable=broad-except
        wait_time = 2**num_attempts
        exception = e
        logging.warning('Waiting %ds to retry... (%s)', wait_time, e)
        time.sleep(2**num_attempts)

    if predictions is None:
      raise ValueError(
          f'Failed to get predictions. ({exception})'
      ) from exception

    if not isinstance(predictions, Iterable):
      predictions = [predictions]

    return [(prediction.text, None) for prediction in predictions]

  def predict_minibatch(
      self, inputs: list[lit_types.JsonDict]
  ) -> list[lit_types.JsonDict]:
    res = [
        {'response': self.query_model(input_dict['prompt'])}
        for input_dict in inputs
    ]
    return res

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        'model_name': lit_types.String(default='gemini-1.0-pro', required=True),
        'max_concurrent_requests': lit_types.Integer(default=4, required=False),
        'max_qps': lit_types.Integer(default=25, required=False),
        'temperature': lit_types.Scalar(default=0.7, required=False),
        'candidate_count': lit_types.Integer(default=1, required=False),
        'max_output_tokens': lit_types.Integer(default=256, required=False),
    }

  def input_spec(self) -> lit_types.Spec:
    return {
        'prompt': lit_types.TextSegment(),
    }

  def output_spec(self) -> lit_types.Spec:
    return {'response': lit_types.GeneratedTextCandidates(parent='prompt')}
