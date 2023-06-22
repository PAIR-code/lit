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

import os

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types
import numpy as np


def get_testdata_path(fname):
  return os.path.join(os.path.dirname(__file__), 'testdata', fname)


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
      ('empty_init', TestDatasetEmptyInit),
      ('pass_thru_init', TestDatasetPassThroughInit),
  )
  def test_init_spec_empty(self, dataset: lit_dataset.Dataset):
    self.assertEmpty(dataset.init_spec())

  def test_init_spec_populated(self):
    self.assertEqual(
        TestDatasetInitWithArgs.init_spec(),
        {
            'path': types.String(),
            'max_examples': types.Integer(default=200, required=False),
            'max_qps': types.Scalar(default=1.0, required=False),
        },
    )

  @parameterized.named_parameters(
      # All base Dataset classes are incompatible with automated spec inference
      # due to the complexity of their arguments, thus return None.
      ('dataset', lit_dataset.Dataset),
      ('indexed_dataset', lit_dataset.IndexedDataset),
      ('none_dataset', lit_dataset.NoneDataset),
  )
  def test_init_spec_none(self, dataset: lit_dataset.Dataset):
    self.assertIsNone(dataset.init_spec())

  def test_remap(self):
    """Test remap method."""
    spec = {
        'score': types.Scalar(),
        'text': types.TextSegment(),
    }
    datapoints = [
        {'score': 0, 'text': 'a'},
        {'score': 0, 'text': 'b'},
    ]
    dset = lit_dataset.Dataset(spec, datapoints)
    remap_dict = {'score': 'val', 'nothing': 'nada'}
    remapped_dset = dset.remap(remap_dict)
    self.assertIn('val', remapped_dset.spec())
    self.assertNotIn('score', remapped_dset.spec())
    self.assertEqual({'val': 0, 'text': 'a'}, remapped_dset.examples[0])


class DatasetLoadingTest(absltest.TestCase):
  """Test to read data from LIT JSONL format."""

  def setUp(self):
    super().setUp()

    self.data_spec = {
        'parity': types.CategoryLabel(vocab=['odd', 'even']),
        'text': types.TextSegment(),
        'value': types.Integer(),
        'other_divisors': types.SparseMultilabel(),
        'in_spanish': types.TextSegment(),
        'embedding': types.Embeddings(),
    }

    self.sample_examples = [
        {
            'parity': 'odd',
            'text': 'One',
            'value': 1,
            'other_divisors': [],
            'in_spanish': 'Uno',
        },
        {
            'parity': 'even',
            'text': 'Two',
            'value': 2,
            'other_divisors': [],
            'in_spanish': 'Dos',
        },
        {
            'parity': 'odd',
            'text': 'Three',
            'value': 3,
            'other_divisors': [],
            'in_spanish': 'Tres',
        },
        {
            'parity': 'even',
            'text': 'Four',
            'value': 4,
            'other_divisors': ['Two'],
            'in_spanish': 'Cuatro',
        },
        {
            'parity': 'odd',
            'text': 'Five',
            'value': 5,
            'other_divisors': [],
            'in_spanish': 'Cinco',
        },
        {
            'parity': 'even',
            'text': 'Six',
            'value': 6,
            'other_divisors': ['Two', 'Three'],
            'in_spanish': 'Seis',
        },
        {
            'parity': 'odd',
            'text': 'Seven',
            'value': 7,
            'other_divisors': [],
            'in_spanish': 'Siete',
        },
        {
            'parity': 'even',
            'text': 'Eight',
            'value': 8,
            'other_divisors': ['Two', 'Four'],
            'in_spanish': 'Ocho',
        },
        {
            'parity': 'odd',
            'text': 'Nine',
            'value': 9,
            'other_divisors': ['Three'],
            'in_spanish': 'Nueve',
        },
        {
            'parity': 'even',
            'text': 'Ten',
            'value': 10,
            'other_divisors': ['Two', 'Five'],
            'in_spanish': 'Diez',
        },
    ]
    # Add embeddings
    rand = np.random.RandomState(42)
    for ex in self.sample_examples:
      vec = rand.normal(0, 1, size=16)
      # Scale such that norm = value, for testing
      scaled = ex['value'] * vec / np.linalg.norm(vec)
      rounded = np.round(scaled, decimals=8)
      # Convert to regular list to avoid issues with assertEqual not correctly
      # handling NumPy array equality.
      ex['embedding'] = rounded.tolist()

    # Index data
    self.indexed_dataset = lit_dataset.IndexedDataset(
        spec=self.data_spec,
        examples=self.sample_examples,
    )

  def test_load_lit_format_unindexed(self):
    ds = lit_dataset.load_lit_format(
        get_testdata_path('count_examples.lit.jsonl')
    )
    self.assertEqual(self.data_spec, ds.spec())
    self.assertEqual(self.sample_examples, ds.examples)

  def test_load_lit_format_indexed(self):
    ds = lit_dataset.load_lit_format(
        get_testdata_path('count_examples.indexed.lit.jsonl'),
    )
    self.assertIsInstance(ds, lit_dataset.IndexedDataset)
    self.assertEqual(self.data_spec, ds.spec())
    self.assertEqual(self.sample_examples, ds.examples)
    self.assertEqual(self.indexed_dataset.indexed_examples, ds.indexed_examples)

  def test_indexed_dataset_load(self):
    ds = self.indexed_dataset.load(
        get_testdata_path('count_examples.indexed.lit.jsonl')
    )
    self.assertIsInstance(ds, lit_dataset.IndexedDataset)
    self.assertEqual(self.data_spec, ds.spec())
    self.assertEqual(self.sample_examples, ds.examples)
    self.assertEqual(self.indexed_dataset.indexed_examples, ds.indexed_examples)

  def test_write_roundtrip(self):
    tempdir = self.create_tempdir()
    output_base = os.path.join(tempdir.full_path, 'test_dataset.lit.jsonl')
    lit_dataset.write_examples(self.sample_examples, output_base)
    lit_dataset.write_spec(self.data_spec, output_base + '.spec')

    # Read back and compare contents
    ds = lit_dataset.load_lit_format(output_base)
    self.assertEqual(self.data_spec, ds.spec())
    self.assertEqual(self.sample_examples, ds.examples)

  def test_write_roundtrip_indexed(self):
    tempdir = self.create_tempdir()
    output_base = os.path.join(
        tempdir.full_path, 'test_dataset.indexed.lit.jsonl'
    )
    lit_dataset.write_examples(
        self.indexed_dataset.indexed_examples, output_base
    )
    lit_dataset.write_spec(self.data_spec, output_base + '.spec')

    # Read back and compare contents
    ds = lit_dataset.load_lit_format(output_base)
    self.assertIsInstance(ds, lit_dataset.IndexedDataset)
    self.assertEqual(self.data_spec, ds.spec())
    self.assertEqual(self.sample_examples, ds.examples)
    self.assertEqual(self.indexed_dataset.indexed_examples, ds.indexed_examples)


if __name__ == '__main__':
  absltest.main()
