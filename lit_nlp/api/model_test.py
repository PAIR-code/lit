# Copyright 2020 Google LLC
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


class _CompatibilityTestModel(model.Model):
  """Dummy model for testing Model.is_compatible_with_dataset()."""

  def __init__(self, input_spec: types.Spec):
    self._input_spec = input_spec

  def input_spec(self) -> types.Spec:
    return self._input_spec

  def output_spec(self) -> types.Spec:
    return {}

  def predict(self, inputs: list[model.JsonDict]) -> list[model.JsonDict]:
    return []


class _BatchingTestModel(model.BatchedModel):
  """A model for testing batched predictions with a minibatch size of 3."""

  def __init__(self):
    self._count = 0

  @property
  def count(self):
    """Returns the number of times predict_minibatch has been called."""
    return self._count

  # LIT API implementation
  def max_minibatch_size(self):
    return 3

  def input_spec(self):
    return {"value": types.Scalar()}

  def output_spec(self):
    return {"scores": types.RegressionScore()}

  def predict_minibatch(self, inputs: list[model.JsonDict], **kw):
    assert len(inputs) <= self.max_minibatch_size()
    self._count += 1
    return map(lambda x: {"scores": x["value"]}, inputs)


class _SavedTestModel(model.Model):
  """A dummy model imitating saved model semantics for testing init_spec()."""

  def __init__(self, path: str, *args, compute_embs: bool = False, **kwargs):
    pass

  def input_spec(self) -> types.Spec:
    return {}

  def output_spec(self) -> types.Spec:
    return {}

  def predict(self, *args, **kwargs) -> list[types.JsonDict]:
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
    ctm = _CompatibilityTestModel(input_spec)
    self.assertEqual(ctm.is_compatible_with_dataset(dataset), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="one_full_batch",
          inputs=[{"value": 1}, {"value": 2}, {"value": 3}],
          expected_outputs=[{"scores": 1}, {"scores": 2}, {"scores": 3}],
          expected_run_count=1,
      ),
      dict(
          testcase_name="one_partial_batch",
          inputs=[{"value": 1}],
          expected_outputs=[{"scores": 1}],
          expected_run_count=1,
      ),
      dict(
          testcase_name="multiple_full_batches",
          inputs=[{"value": 1}, {"value": 2}, {"value": 3},
                  {"value": 4}, {"value": 5}, {"value": 6}],
          expected_outputs=[{"scores": 1}, {"scores": 2}, {"scores": 3},
                            {"scores": 4}, {"scores": 5}, {"scores": 6}],
          expected_run_count=2,
      ),
      dict(
          testcase_name="multiple_partial_batches",
          inputs=[{"value": 1}, {"value": 2}, {"value": 3},
                  {"value": 4}],
          expected_outputs=[{"scores": 1}, {"scores": 2}, {"scores": 3},
                            {"scores": 4}],
          expected_run_count=2,
      ),
  )
  def test_batched_predict(self, inputs: list[model.JsonDict],
                           expected_outputs: list[model.JsonDict],
                           expected_run_count: int):
    """Tests predict() for a model with a batch size of 3."""
    # Note that TestModelBatched
    test_model = _BatchingTestModel()
    result = list(test_model.predict(inputs))
    self.assertListEqual(result, expected_outputs)
    self.assertEqual(len(result), len(inputs))
    self.assertEqual(test_model.count, expected_run_count)

  def test_init_spec_empty(self):
    self.assertEmpty(_BatchingTestModel.init_spec())

  def test_init_spec_populated(self):
    self.assertEqual(
        _SavedTestModel.init_spec(),
        {
            "path": types.String(),
            "compute_embs": types.Boolean(default=False, required=False),
        },
    )

  @parameterized.named_parameters(
      ("bad_args", _CompatibilityTestModel),
      # All ModelWrapper instances should return None, regardless of the model
      # the instance is wrapping.
      ("wrapper", model.ModelWrapper),
  )
  def test_init_spec_none(self, mdl: model.Model):
    self.assertIsNone(mdl.init_spec())


if __name__ == "__main__":
  absltest.main()
