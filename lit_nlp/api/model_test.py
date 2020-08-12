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
# Lint as: python3
"""Tests for lit_nlp.lib.model."""

from absl.testing import absltest

from lit_nlp.api import model
from lit_nlp.api import types
from lit_nlp.lib import testing_utils


class SpecTest(absltest.TestCase):

  def test_compatibility_fullmatch(self):
    """Test with an exact match."""
    mspec = model.ModelSpec(
        input={
            "text_a": types.TextSegment(),
            "text_b": types.TextSegment(),
        },
        output={})
    dspec = mspec.input
    self.assertTrue(mspec.is_compatible_with_dataset(dspec))

  def test_compatibility_mismatch(self):
    """Test with specs that don't match."""
    mspec = model.ModelSpec(
        input={
            "text_a": types.TextSegment(),
            "text_b": types.TextSegment(),
        },
        output={})
    dspec = {"premise": types.TextSegment(), "hypothesis": types.TextSegment()}
    self.assertFalse(mspec.is_compatible_with_dataset(dspec))

  def test_compatibility_extrafield(self):
    """Test with an extra field in the dataset."""
    mspec = model.ModelSpec(
        input={
            "text_a": types.TextSegment(),
            "text_b": types.TextSegment(),
        },
        output={})
    dspec = {
        "text_a": types.TextSegment(),
        "text_b": types.TextSegment(),
        "label": types.CategoryLabel(vocab=["0", "1"]),
    }
    self.assertTrue(mspec.is_compatible_with_dataset(dspec))

  def test_compatibility_optionals(self):
    """Test with optionals in the model spec."""
    mspec = model.ModelSpec(
        input={
            "text": types.TextSegment(),
            "tokens": types.Tokens(parent="text", required=False),
            "label": types.CategoryLabel(vocab=["0", "1"], required=False),
        },
        output={})
    dspec = {
        "text": types.TextSegment(),
        "label": types.CategoryLabel(vocab=["0", "1"]),
    }
    self.assertTrue(mspec.is_compatible_with_dataset(dspec))

  def test_compatibility_optionals_mismatch(self):
    """Test with optionals that don't match metadata."""
    mspec = model.ModelSpec(
        input={
            "text": types.TextSegment(),
            "tokens": types.Tokens(parent="text", required=False),
            "label": types.CategoryLabel(vocab=["0", "1"], required=False),
        },
        output={})
    dspec = {
        "text": types.TextSegment(),
        # This label field doesn't match the one the model expects.
        "label": types.CategoryLabel(vocab=["foo", "bar"]),
    }
    self.assertFalse(mspec.is_compatible_with_dataset(dspec))


class ModelTest(absltest.TestCase):

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
