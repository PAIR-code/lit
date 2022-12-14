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
"""Tests for lit_nlp.lib.model."""

from absl.testing import absltest
from absl.testing import parameterized

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model
from lit_nlp.api import types
from lit_nlp.lib import testing_utils


class CompatibilityTestModel(model.Model):
  """Dummy model for testing Model.is_compatible_with_dataset()."""

  def __init__(self, input_spec: types.Spec):
    self._input_spec = input_spec

  def input_spec(self) -> types.Spec:
    return self._input_spec

  def output_spec(self) -> types.Spec:
    return {}

  def predict_minibatch(self,
                        inputs: list[model.JsonDict]) -> list[model.JsonDict]:
    return []


class ModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="full_match",
          input_spec={
              "text_a": types.TextSegment(),
              "text_b": types.TextSegment(),
          },
          dataset_spec={
              "text_a": types.TextSegment(),
              "text_b": types.TextSegment(),
          },
          expected=True,
      ),
      dict(
          testcase_name="mismatch",
          input_spec={
              "text_a": types.TextSegment(),
              "text_b": types.TextSegment(),
          },
          dataset_spec={
              "premise": types.TextSegment(),
              "hypothesis": types.TextSegment(),
          },
          expected=False,
      ),
      dict(
          testcase_name="extra_field",
          input_spec={
              "text_a": types.TextSegment(),
              "text_b": types.TextSegment(),
          },
          dataset_spec={
              "text_a": types.TextSegment(),
              "text_b": types.TextSegment(),
              "label": types.CategoryLabel(vocab=["0", "1"]),
          },
          expected=True,
      ),
      dict(
          testcase_name="optionals",
          input_spec={
              "text": types.TextSegment(),
              "tokens": types.Tokens(parent="text", required=False),
              "label": types.CategoryLabel(vocab=["0", "1"], required=False),
          },
          dataset_spec={
              "text": types.TextSegment(),
              "label": types.CategoryLabel(vocab=["0", "1"]),
          },
          expected=True,
      ),
      dict(
          testcase_name="optionals_mismatch",
          input_spec={
              "text": types.TextSegment(),
              "tokens": types.Tokens(parent="text", required=False),
              "label": types.CategoryLabel(vocab=["0", "1"], required=False),
          },
          dataset_spec={
              "text": types.TextSegment(),
              # This label field doesn't match the one the model expects.
              "label": types.CategoryLabel(vocab=["foo", "bar"]),
          },
          expected=False,
      ),
  )
  def test_compatibility(self, input_spec: types.Spec, dataset_spec: types.Spec,
                         expected: bool):
    """Test spec compatibility between models and datasets."""
    dataset = lit_dataset.Dataset(spec=dataset_spec)
    ctm = CompatibilityTestModel(input_spec)
    self.assertEqual(ctm.is_compatible_with_dataset(dataset), expected)

  def test_predict(self):
    """Tests predict() for a model with max batch size of 3."""

    # Input of less than 1 batch.
    test_model = testing_utils.TestModelBatched()
    result = test_model.predict([{"value": 1}, {"value": 2}])

    self.assertListEqual(list(result), [{"scores": 1},
                                        {"scores": 2}])
    self.assertEqual(test_model.count, 1)

    # Between 1 and 2 full batches.
    test_model = testing_utils.TestModelBatched()
    result = test_model.predict([{"value": 1}, {"value": 2}, {"value": 3},
                                 {"value": 4}])

    self.assertListEqual(list(result), [{"scores": 1},
                                        {"scores": 2},
                                        {"scores": 3},
                                        {"scores": 4}])
    self.assertEqual(test_model.count, 2)

    # Input of 2 full batches.
    test_model = testing_utils.TestModelBatched()
    result = test_model.predict([{"value": 1}, {"value": 2}, {"value": 3},
                                 {"value": 4}, {"value": 5}, {"value": 6}])

    self.assertListEqual(list(result), [{"scores": 1},
                                        {"scores": 2},
                                        {"scores": 3},
                                        {"scores": 4},
                                        {"scores": 5},
                                        {"scores": 6}])
    self.assertEqual(test_model.count, 2)


if __name__ == "__main__":
  absltest.main()
