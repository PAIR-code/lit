# Copyright 2020 Google LLC
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
"""Testing utilities for python backend.

Contains things like dummy LIT Model so we don't have to define it in every
test.
"""

from collections.abc import Iterable, Iterator

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
import numpy as np
import numpy.testing as npt

JsonDict = lit_types.JsonDict


class RegressionModelForTesting(lit_model.BatchedModel):
  """Implements lit.Model interface for testing.

  This class allows flexible input spec to allow different testing scenarios.
  """

  def __init__(self, input_spec: lit_types.Spec):
    """Set input spec.

    Args:
      input_spec: An input spec.
    """
    self._input_spec = input_spec

  # LIT API implementation
  def input_spec(self):
    return self._input_spec

  def output_spec(self):
    return {'scores': lit_types.RegressionScore(parent='label')}

  def predict_minibatch(self, inputs: list[JsonDict], **kw):
    return self.predict(inputs)

  def predict(self, inputs: Iterable[JsonDict], *args,
              **kw) -> Iterator[JsonDict]:
    """Return 0.0 regression values for all examples.

    Args:
      inputs: input examples
      *args: unused
      **kw: unused

    Returns:
      predictions
    """
    return map(lambda x: {'scores': 0.0}, inputs)


class IdentityRegressionModelForTesting(lit_model.BatchedModel):
  """Implements lit.Model interface for testing.

  This class reflects the input in the prediction for simple testing.
  """

  def __init__(self):
    self._count = 0

  def input_spec(self):
    return {'val': lit_types.Scalar()}

  def output_spec(self):
    return {'score': lit_types.RegressionScore()}

  def predict_minibatch(self, inputs: list[JsonDict], **kw):
    return self.predict(inputs)

  def predict(self, inputs: Iterable[JsonDict], *args,
              **kw) -> Iterator[JsonDict]:
    """Return input value for all examples.

    Args:
      inputs: input examples
      *args: unused
      **kw: unused

    Returns:
      predictions
    """
    results = [{'score': input['val']} for input in inputs]
    self._count += len(results)
    return iter(results)

  @property
  def count(self):
    """Returns the number of times predict has been called."""
    return self._count


class ClassificationModelForTesting(lit_model.BatchedModel):
  """Implements lit.Model interface for testing classification models.

     Returns the same output for every input.
  """

  # LIT API implementation
  def input_spec(self):
    return {'input_embs': lit_types.TokenEmbeddings(align='tokens',
                                                    required=False),
            'segment': lit_types.TextSegment(),
            'grad_class': lit_types.CategoryLabel(vocab=['0', '1'])}

  def output_spec(self):
    return {
        'probas':
            lit_types.MulticlassPreds(
                parent='label', vocab=['0', '1'], null_idx=0),
        'input_embs':
            lit_types.TokenEmbeddings(align='tokens'),
        'input_embs_grad':
            lit_types.TokenGradients(
                align='tokens',
                grad_for='input_embs',
                grad_target_field_key='grad_class'),
        'tokens':
            lit_types.Tokens(),
        'grad_class':
            lit_types.CategoryLabel(vocab=['0', '1'])
    }

  def predict_minibatch(self, inputs: list[JsonDict], **kw):
    output = {
        'probas': np.array([0.2, 0.8]),
        'input_embs': np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        ),
        'input_embs_grad': np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        ),
        'tokens': ['test'],
        'grad_class': '1',
    }
    return map(lambda x: output, inputs)


def fake_projection_input(n, num_dims):
  """Generates random embeddings in the correct format."""
  rng = np.random.RandomState(42)
  return [{'x': rng.rand(num_dims)} for _ in range(n)]


def assert_deep_almost_equal(testcase, result, actual, places=4):
  """Checks if provided inputs are almost equal, recurses on dicts values."""
  if isinstance(result, (int, float)):
    testcase.assertAlmostEqual(result, actual, places=places)
  elif isinstance(result, (list)):
    if all(isinstance(n, (int, float)) for n in result):
      rtol = 10 ** (-1 * places)
      npt.assert_allclose(result, actual, rtol=rtol, atol=1e-4)
    elif all(isinstance(n, dict) for n in result):
      for i in range(len(result)):
        assert_deep_almost_equal(testcase, result[i], actual[i])
  elif isinstance(result, dict):
    if set(result.keys()) != set(actual.keys()):
      testcase.fail('results and actual have different keys')
    for key in result:
      assert_deep_almost_equal(testcase, result[key], actual[key])


class CustomOutputModelForTesting(lit_model.BatchedModel):
  """Implements lit.Model interface for testing.

  This class allows user-specified outputs for testing return values.
  """

  def __init__(
      self,
      input_spec: lit_types.Spec,
      output_spec: lit_types.Spec,
      results: list[JsonDict],
  ):
    """Set model internals.

    Args:
      input_spec: An input spec.
      output_spec: An output spec.
      results: Results to return.
    """
    self._input_spec = input_spec
    self._output_spec = output_spec
    self._predict_counter = 0
    self._results = results

  # LIT API implementation
  def input_spec(self):
    return self._input_spec

  def output_spec(self):
    return self._output_spec

  def predict_minibatch(self, inputs: list[JsonDict], **kw):
    def predict_single(_):
      output = self._results[self._predict_counter % len(self._results)]
      self._predict_counter += 1
      return output
    return map(predict_single, inputs)
