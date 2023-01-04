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
from lit_nlp.api import types


class TestDatasetEmptyInit(lit_dataset.Dataset):

  def __init__(self):
    pass


class TestDatasetPassThroughInit(lit_dataset.Dataset):

  def __init__(self, *args, **kwargs):
    pass


class TestDatasetInitWithArgs(lit_dataset.Dataset):

  def __init__(self, path: str, max_examples: int = 200, max_qps: float = 1.0):
    pass


class DatasetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty_init", TestDatasetEmptyInit()),
      ("pass_thru_init", TestDatasetPassThroughInit()),
  )
  def test_init_spec_empty(self, dataset: lit_dataset.Dataset):
    self.assertEmpty(dataset.init_spec())

  def test_init_spec_populated(self):
    dataset = TestDatasetInitWithArgs("test/path")
    self.assertEqual(dataset.init_spec(), {
        "path": types.String(),
        "max_examples": types.Integer(default=200, required=False),
        "max_qps": types.Scalar(default=1.0, required=False),
    })

  @parameterized.named_parameters(
      # All base Dataset classes are incompatible with automated spec inference
      # due to the complexity of their arguments, thus return None.
      ("dataset", lit_dataset.Dataset()),
      ("indexed_dataset", lit_dataset.IndexedDataset(id_fn=lambda x: x)),
      ("none_dataset", lit_dataset.NoneDataset(models={})),
  )
  def test_init_spec_none(self, dataset: lit_dataset.Dataset):
    self.assertIsNone(dataset.init_spec())

  def test_remap(self):
    """Test remap method."""
    spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    datapoints = [
        {
            "score": 0,
            "text": "a"
        },
        {
            "score": 0,
            "text": "b"
        },
    ]
    dset = lit_dataset.Dataset(spec, datapoints)
    remap_dict = {"score": "val", "nothing": "nada"}
    remapped_dset = dset.remap(remap_dict)
    self.assertIn("val", remapped_dset.spec())
    self.assertNotIn("score", remapped_dset.spec())
    self.assertEqual({"val": 0, "text": "a"}, remapped_dset.examples[0])


if __name__ == "__main__":
  absltest.main()
