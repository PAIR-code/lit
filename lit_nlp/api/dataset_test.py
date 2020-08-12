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
from lit_nlp.api import types


class DatasetTest(absltest.TestCase):

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
