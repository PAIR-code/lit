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

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.components import static_preds


class StaticPredictionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    number_names = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine"
    ]
    # pylint: disable=g-complex-comprehension
    dummy_inputs = [{
        "name": name,
        "val": i
    } for i, name in enumerate(number_names)]
    # pylint: enable=g-complex-comprehension
    input_spec = {"name": lit_types.TextSegment(), "val": lit_types.Scalar()}
    self.input_ds = lit_dataset.Dataset(input_spec, dummy_inputs)

    dummy_preds = [{
        "pred": "{name:s}={val:d}".format(**d)
    } for d in self.input_ds.examples]
    self.preds_ds = lit_dataset.Dataset({"pred": lit_types.TextSegment()},
                                        dummy_preds)

  def test_all_identifiers(self):
    """Test using all fields as identifier keys."""
    model = static_preds.StaticPredictions(
        self.input_ds, self.preds_ds, input_identifier_keys=["name", "val"])
    self.assertEqual(self.input_ds.spec(), model.input_spec())
    self.assertEqual(self.preds_ds.spec(), model.output_spec())
    # Test on whole dataset
    self.assertEqual(
        list(model.predict(self.input_ds.examples)),
        list(self.preds_ds.examples))
    # Test on a slice
    self.assertEqual(
        list(model.predict(self.input_ds.examples[2:5])),
        list(self.preds_ds.examples[2:5]))
    # Test on unknown examples
    with self.assertRaises(KeyError):
      # Should raise an exception on the second input.
      inputs = [{"name": "nine", "val": 9}, {"name": "twenty", "val": 20}]
      # Use list() to force the generator to run.
      _ = list(model.predict(inputs))

  def test_partial_identifiers(self):
    """Test using only some fields as identifier keys."""
    model = static_preds.StaticPredictions(
        self.input_ds, self.preds_ds, input_identifier_keys=["name"])
    self.assertEqual({"name": lit_types.TextSegment()}, model.input_spec())
    self.assertEqual(self.preds_ds.spec(), model.output_spec())
    # Test on whole dataset
    self.assertEqual(
        list(model.predict(self.input_ds.examples)),
        list(self.preds_ds.examples))
    # Test on a slice
    self.assertEqual(
        list(model.predict(self.input_ds.examples[2:5])),
        list(self.preds_ds.examples[2:5]))
    # Test on unknown examples
    with self.assertRaises(KeyError):
      # Should raise an exception on the second input.
      inputs = [{"name": "nine", "val": 9}, {"name": "twenty", "val": 20}]
      # Use list() to force the generator to run.
      _ = list(model.predict(inputs))


if __name__ == "__main__":
  absltest.main()
