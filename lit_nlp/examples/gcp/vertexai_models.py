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

"""Model Wrapper for generative models."""

from collections.abc import Iterable
import logging
import time
from typing import Optional, Union
from google.cloud import aiplatform
from vertexai import generative_models
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types

_MAX_NUM_RETRIES = 5
_DEFAULT_CANDIDATE_COUNT = 1
_DEFAULT_MAX_OUTPUT_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.7


class GeminiFoundationalModel(lit_model.BatchedRemoteModel):
  """GeminiFoundationalModel is a wrapper for foundatoinal Gemini models in Vertex AI Model Garden model.

  Attributes:
    model_name: The name of the model to load.
    max_concurrent_requests: The maximum number of concurrent requests to the
      model.
    max_qps: The maximum number of queries per second to the model.
    temperature: The temperature to use for the model.
    max_output_tokens: The maximum number of tokens to generate.

  Please note the model will predict all examples at a fixed temperature.
  """

  def __init__(
      self,
      model_name: str,
      max_concurrent_requests: int = 4,
      max_qps: Union[int, float] = 25,
      temperature: Optional[float] = _DEFAULT_TEMPERATURE,
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
        'temperature': lit_types.Scalar(
            default=_DEFAULT_TEMPERATURE, required=False
        ),
        'candidate_count': lit_types.Integer(default=1, required=False),
        'max_output_tokens': lit_types.Integer(default=256, required=False),
    }

  def input_spec(self) -> lit_types.Spec:
    return {
        'prompt': lit_types.TextSegment(),
    }

  def output_spec(self) -> lit_types.Spec:
    return {'response': lit_types.GeneratedTextCandidates(parent='prompt')}


class SelfHostedGenerativeModel(lit_model.BatchedRemoteModel):
  """SelfHostedGenerativeModel is a wrapper for self-hosted generative models.

  This model wrapper is used for self-hosted generative models that require
  self-deployment.
  The model deployment process is managed by the user, and described in
  https://cloud.google.com/vertex-ai/docs/pipelines/model-endpoint-component. It
  is recommended deploy the model in Vertex AI. After the model is deployed,
  an aip_endpoint_name will be provided, and can be used to query the
  model.

  Attributes:
    aip_endpoint_name: A fully-qualified VertexAI depolyed model endpoint
      resource name or endpoint ID.
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
      aip_endpoint_name: str,
      max_concurrent_requests: int = 4,
      max_qps: Union[int, float] = 25,
      temperature: Optional[float] = _DEFAULT_TEMPERATURE,
      max_output_tokens: Optional[int] = _DEFAULT_MAX_OUTPUT_TOKENS,
  ):
    super().__init__(
        max_concurrent_requests=max_concurrent_requests, max_qps=max_qps
    )
    self.temperature = temperature
    self.max_output_tokens = max_output_tokens
    self._endpoint = aiplatform.models.Endpoint(aip_endpoint_name)

  def query_model(self, input_text: str) -> lit_types.ScoredTextCandidates:
    num_attempts = 0
    predictions = None
    exception = None

    instances = [
        {
            'prompt': input_text,
            'max_tokens': self.max_output_tokens,
            'temperature': self.temperature,
        },
    ]

    while num_attempts < _MAX_NUM_RETRIES and predictions is None:
      num_attempts += 1

      try:
        predictions = self._endpoint.predict(instances).predictions
      except Exception as e:  # pylint: disable=broad-except
        wait_time = 2**num_attempts
        exception = e
        logging.warning('Waiting %ds to retry... (%s)', wait_time, e)
        time.sleep(2**num_attempts)

    if predictions is None:
      raise ValueError(
          'Failed to get predictions with endpoint %s, after %d attempts.'
          % (self._endpoint.name, _MAX_NUM_RETRIES)
      ) from exception

    if not isinstance(predictions, Iterable):
      predictions = [predictions]

    return [(prediction, None) for prediction in predictions]

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
        'aip_endpoint_name': lit_types.String(default='', required=True),
        'max_concurrent_requests': lit_types.Integer(default=4, required=False),
        'max_qps': lit_types.Integer(default=25, required=False),
        'temperature': lit_types.Scalar(
            default=_DEFAULT_TEMPERATURE, required=False
        ),
        'max_output_tokens': lit_types.Integer(default=256, required=False),
    }

  def input_spec(self) -> lit_types.Spec:
    return {
        'prompt': lit_types.TextSegment(),
    }

  def output_spec(self) -> lit_types.Spec:
    return {'response': lit_types.GeneratedTextCandidates(parent='prompt')}
