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
import types

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import numpy as np


def get_testdata_path(fname):
  return os.path.join(os.path.dirname(__file__), 'testdata', fname)


class _EmptyInitTestDataset(lit_dataset.Dataset):

  def __init__(self):
    pass


class _PassThroughInitTestDataset(lit_dataset.Dataset):

  def __init__(self, *args, **kwargs):
    pass


class _InitWithArgsTestDataset(lit_dataset.Dataset):

  def __init__(self, path: str, max_examples: int = 200, max_qps: float = 1.0):
    pass


class DatasetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty_init', _EmptyInitTestDataset),
      ('pass_thru_init', _PassThroughInitTestDataset),
  )
  def test_init_spec_empty(self, dataset: lit_dataset.Dataset):
    self.assertEmpty(dataset.init_spec())

  def test_init_spec_populated(self):
    self.assertEqual(
        _InitWithArgsTestDataset.init_spec(),
        {
            'path': lit_types.String(),
            'max_examples': lit_types.Integer(default=200, required=False),
            'max_qps': lit_types.Scalar(default=1.0, required=False),
        },
    )

  @parameterized.named_parameters(
      # All base Dataset classes are incompatible with automated spec inference
      # due to the complexity of their arguments, thus return None.
      ('dataset', lit_dataset.Dataset),
      ('none_dataset', lit_dataset.NoneDataset),
  )
  def test_init_spec_none(self, dataset: lit_dataset.Dataset):
    self.assertIsNone(dataset.init_spec())

  def test_remap(self):
    """Test remap method."""
    spec = {
        'score': lit_types.Scalar(),
        'text': lit_types.TextSegment(),
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


class InputHashTest(parameterized.TestCase):
  """Test to hash data correctly, not using _id or _meta fields."""

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_example',
          example={},
          expected_hash='99914b932bd37a50b983c5e7c90ae93b',
      ),
      dict(
          testcase_name='one_field_example',
          example={'value': 1},
          expected_hash='1ff00094a5ba112cb7dd128e783d6803',
      ),
      dict(
          testcase_name='three_field_example',
          example={
              'parity': 'odd',
              'text': 'One',
              'value': 1,
          },
          expected_hash='25dd56cf3b51e8e2954575f88b2620ca',
      ),
      dict(
          testcase_name='has_id_field',
          example={
              'parity': 'odd',
              'text': 'One',
              'value': 1,
              '_id': 'some_random_id',
          },
          expected_hash='25dd56cf3b51e8e2954575f88b2620ca',
      ),
      dict(
          testcase_name='has_meta_field',
          example={
              'parity': 'odd',
              'text': 'One',
              'value': 1,
              '_meta': lit_types.InputMetadata(
                  added=None, parentId=None, source=None
              ),
          },
          expected_hash='25dd56cf3b51e8e2954575f88b2620ca',
      ),
      dict(
          testcase_name='has_id_and_meta_fields',
          example={
              'parity': 'odd',
              'text': 'One',
              'value': 1,
              '_id': 'some_random_id',
              '_meta': lit_types.InputMetadata(
                  added=None, parentId=None, source=None
              ),
          },
          expected_hash='25dd56cf3b51e8e2954575f88b2620ca',
      ),
  )
  def test_hash(
      self, example: lit_types.Input, expected_hash: lit_types.ExampleId
  ):
    input_hash = lit_dataset.input_hash(example)
    self.assertEqual(input_hash, expected_hash)


class IndexedDatasetTest(absltest.TestCase):

  _DATASET_SPEC: lit_types.Spec = {
      'parity': lit_types.CategoryLabel(vocab=['odd', 'even']),
      'text': lit_types.TextSegment(),
      'value': lit_types.Integer(),
  }

  def test_init_from_examples_without_ids(self):
    # This test ensures that IDs are computed and assigned to IndexedInput.id
    # and IndexedInput.data._id fields for examples wihtout IDs.
    examples: list[lit_types.JsonDict] = [
        {'parity': 'odd', 'text': 'one', 'value': 1},
        {'parity': 'even', 'text': 'two', 'value': 2},
        {'parity': 'odd', 'text': 'three', 'value': 3},
    ]
    dataset = lit_dataset.IndexedDataset(
        spec=self._DATASET_SPEC,
        examples=examples
    )

    # TODO(b/266681945): Enabled zip(..., strict=true) once updated to Py3.10
    for indexed_example, example, original in zip(
        dataset.indexed_examples, dataset.examples, examples
    ):
      self.assertIsInstance(example['_id'], str)
      self.assertEqual(indexed_example['id'], example['_id'])

      self.assertIsNotNone(example['_meta'])
      self.assertEqual(indexed_example['meta'], example['_meta'])

      self.assertEqual(indexed_example['data'], example)

      for key in original:
        self.assertEqual(example[key], original[key])

  def test_init_from_examples_with_ids(self):
    # This test ensures that IndexedInput.id is the same as
    # IndexedInput.data._id when initialized from examples with _id fields.
    examples: list[lit_types.JsonDict] = [
        {'parity': 'odd', 'text': 'one', 'value': 1, '_id': 'one-1-odd'},
        {'parity': 'even', 'text': 'two', 'value': 2, '_id': 'two-2-even'},
        {'parity': 'odd', 'text': 'three', 'value': 3, '_id': 'three-3-odd'},
    ]
    dataset = lit_dataset.IndexedDataset(
        spec=self._DATASET_SPEC,
        examples=examples
    )

    for indexed_example, example, original in zip(
        dataset.indexed_examples, dataset.examples, examples
    ):
      self.assertEqual(example['_id'], original['_id'])
      self.assertEqual(indexed_example['id'], original['_id'])

      self.assertIsNotNone(example['_meta'])
      self.assertEqual(indexed_example['meta'], example['_meta'])

      self.assertEqual(indexed_example['data'], example)

      for key in original:
        self.assertEqual(example[key], original[key])

  def test_init_from_indexed_examples_with_ids(self):
    # This tests initializing from fully compliant IndexedInputs.
    indexed_examples: list[lit_types.IndexedInput] = [
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'odd',
                'text': 'one',
                'value': 1,
                '_id': 'one-1-odd',
            }),
            id='one-1-odd',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'even',
                'text': 'two',
                'value': 2,
                '_id': 'two-2-even',
            }),
            id='two-2-even',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'odd',
                'text': 'three',
                'value': 3,
                '_id': 'three-3-odd',
            }),
            id='three-3-odd',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
    ]

    dataset = lit_dataset.IndexedDataset(
        spec=self._DATASET_SPEC,
        indexed_examples=indexed_examples
    )

    for indexed_example, example, original in zip(
        dataset.indexed_examples, dataset.examples, indexed_examples
    ):
      self.assertEqual(example['_id'], original['id'])
      self.assertEqual(indexed_example['id'], original['id'])

      self.assertEqual(indexed_example['meta'], example['_meta'])

      for key in (original_data := original['data']):
        self.assertEqual(example[key], original_data[key])
        self.assertEqual(indexed_example['data'][key], original_data[key])

  def test_init_from_indexed_examples_without_ids(self):
    # This test represents the case where Legacy LIT saved data has been
    # correctly initialized in an IndexedInput with a readonly view of the
    # IndexedInput.data property, but the IndexedInput.data does not have an
    # _id property.
    indexed_examples: list[lit_types.IndexedInput] = [
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'odd',
                'text': 'one',
                'value': 1,
            }),
            id='one-1-odd',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'even',
                'text': 'two',
                'value': 2,
            }),
            id='two-2-even',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'odd',
                'text': 'three',
                'value': 3,
            }),
            id='three-3-odd',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
    ]

    dataset = lit_dataset.IndexedDataset(
        spec=self._DATASET_SPEC,
        indexed_examples=indexed_examples
    )

    for indexed_example, example, original in zip(
        dataset.indexed_examples, dataset.examples, indexed_examples
    ):
      self.assertEqual(example['_id'], original['id'])
      self.assertEqual(indexed_example['id'], original['id'])

      self.assertEqual(indexed_example['meta'], example['_meta'])

      for key in (original_data := original['data']):
        self.assertEqual(example[key], original_data[key])
        self.assertEqual(indexed_example['data'][key], original_data[key])

  def test_init_from_pesudo_indexed_examples(self):
    # This test represents the case where Legacy LIT saved data is loaded via
    # load_lit_format() or some other some external process where the examples
    # are merely cast to IndexedInput. It ensure that all IndexedInput.data and
    # JsonDict example representations are readonly via MappingProxyType.
    indexed_examples = [
        {
            'data': {
                'parity': 'odd',
                'text': 'one',
                'value': 1,
            },
            'id': 'one-1-odd',
            'meta': {},
        },
        {
            'data': {
                'parity': 'even',
                'text': 'two',
                'value': 2,
            },
            'id': 'two-2-even',
            'meta': {},
        },
        {
            'data': {
                'parity': 'odd',
                'text': 'three',
                'value': 3,
            },
            'id': 'three-3-odd',
            'meta': {},
        },
    ]

    dataset = lit_dataset.IndexedDataset(
        spec=self._DATASET_SPEC,
        indexed_examples=indexed_examples
    )

    for indexed_example, example, original in zip(
        dataset.indexed_examples, dataset.examples, indexed_examples
    ):
      self.assertEqual(example['_id'], original['id'])
      self.assertEqual(indexed_example['id'], original['id'])

      self.assertEqual(indexed_example['meta'], example['_meta'])

      indexed_example_data = indexed_example['data']
      self.assertIsInstance(example, types.MappingProxyType)
      self.assertIsInstance(indexed_example_data, types.MappingProxyType)
      self.assertEqual(example, indexed_example_data)
      for key in (original_data := original['data']):
        self.assertEqual(example[key], original_data[key])
        self.assertEqual(indexed_example_data[key], original_data[key])

  def test_init_from_indexed_examples_with_inconsistent_ids(self):
    # This test ensures that the IndexedInput.id property supersedes the
    # JsonDict._id property if both are provided but their values are different.
    indexed_examples: list[lit_types.IndexedInput] = [
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'odd',
                'text': 'one',
                'value': 1,
                '_id': 'odd-1-one',
            }),
            id='one-1-odd',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'even',
                'text': 'two',
                'value': 2,
                '_id': 'even-2-two',
            }),
            id='two-2-even',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
        lit_types.IndexedInput(
            data=types.MappingProxyType({
                'parity': 'odd',
                'text': 'three',
                'value': 3,
                '_id': 'odd-3-three',
            }),
            id='three-3-odd',
            meta=lit_types.InputMetadata(
                added=None, parentId=None, source=None
            ),
        ),
    ]

    dataset = lit_dataset.IndexedDataset(
        spec=self._DATASET_SPEC,
        indexed_examples=indexed_examples
    )

    for indexed_example, example, original in zip(
        dataset.indexed_examples, dataset.examples, indexed_examples
    ):
      self.assertEqual(example['_id'], original['id'])
      self.assertEqual(indexed_example['id'], original['id'])
      self.assertEqual(indexed_example['meta'], example['_meta'])

      original_data = original['data']
      self.assertNotEqual(example['_id'], original_data['_id'])
      self.assertNotEqual(indexed_example['id'], original_data['_id'])

      for key in dataset.spec().keys():
        self.assertEqual(example[key], original_data[key])
        self.assertEqual(indexed_example['data'][key], original_data[key])

  def test_init_without_examples(self):
    dataset = lit_dataset.IndexedDataset(spec=self._DATASET_SPEC)
    self.assertEmpty(dataset.examples)
    self.assertEmpty(dataset.indexed_examples)
    self.assertEqual(dataset.spec(), self._DATASET_SPEC)
    self.assertIsNone(dataset.init_spec())


class DatasetLoadingTest(absltest.TestCase):
  """Test to read data from LIT JSONL format."""

  def setUp(self):
    super().setUp()

    self.data_spec = {
        'parity': lit_types.CategoryLabel(vocab=['odd', 'even']),
        'text': lit_types.TextSegment(),
        'value': lit_types.Integer(),
        'other_divisors': lit_types.SparseMultilabel(),
        'in_spanish': lit_types.TextSegment(),
        'embedding': lit_types.Embeddings(),
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
    self.assertEqual(self.indexed_dataset.indexed_examples, ds.indexed_examples)
    for original, loaded in zip(self.sample_examples, ds.examples):
      for key in self.indexed_dataset.spec().keys():
        self.assertEqual(original[key], loaded[key])

  def test_indexed_dataset_load(self):
    ds = self.indexed_dataset.load(
        get_testdata_path('count_examples.indexed.lit.jsonl')
    )
    self.assertIsInstance(ds, lit_dataset.IndexedDataset)
    self.assertEqual(self.data_spec, ds.spec())
    self.assertEqual(self.indexed_dataset.indexed_examples, ds.indexed_examples)
    for original, loaded in zip(self.sample_examples, ds.examples):
      for key in self.indexed_dataset.spec().keys():
        self.assertEqual(original[key], loaded[key])

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
    self.assertEqual(self.indexed_dataset.indexed_examples, ds.indexed_examples)
    for original, loaded in zip(self.sample_examples, ds.examples):
      for key in self.indexed_dataset.spec().keys():
        self.assertEqual(original[key], loaded[key])


if __name__ == '__main__':
  absltest.main()
