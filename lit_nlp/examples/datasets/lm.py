"""Language modeling datasets."""

import copy
import json
import os
import glob
from typing import Optional

from absl import logging
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import tensorflow_datasets as tfds

SAMPLE_DATA_DIR = os.path.dirname(__file__)


class PlaintextSents(lit_dataset.Dataset):
  """Load sentences from a flat text file."""

  def __init__(
      self,
      path_or_glob: str,
      skiplines: int = 0,
      max_examples: Optional[int] = None,
      field_name: str = 'text',
  ):
    self.field_name = field_name
    self._examples = self.load_datapoints(path_or_glob, skiplines=skiplines)[
        :max_examples
    ]

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    default_path = ''

    return {
        'path_or_glob': lit_types.String(
            default=default_path, required=False
        ),
        'skiplines': lit_types.Integer(default=0, max_val=25),
        'max_examples': lit_types.Integer(
            default=1000, min_val=0, max_val=10_000, required=False
        ),
    }

  def load_datapoints(self, path_or_glob: str, skiplines: int = 0):
    examples = []
    for path in glob.glob(path_or_glob):
      with open(path) as fd:
        for i, line in enumerate(fd):
          if i < skiplines:  # skip header lines, if necessary
            continue
          line = line.strip()
          if line:  # skip blank lines, these are usually document breaks
            examples.append({self.field_name: line})
    return examples

  def load(self, path: str):
    return lit_dataset.Dataset(base=self, examples=self.load_datapoints(path))

  def spec(self) -> lit_types.Spec:
    """Should match MLM's input_spec()."""
    return {self.field_name: lit_types.TextSegment()}


class PromptExamples(lit_dataset.Dataset):
  """Prompt examples for modern LMs."""

  SAMPLE_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, 'prompt_examples.jsonl')

  def load_datapoints(self, path: str):
    if not path:
      logging.warn(
          'Empty path to PromptExamples.load_datapoints(). Returning empty'
          ' dataset.'
      )
      return []

    default_ex_values = {
        k: copy.deepcopy(field_spec.default)
        for k, field_spec in self.spec().items()
    }

    examples = []
    with open(path) as fd:
      for line in fd:
        examples.append(default_ex_values | json.loads(line))

    return examples

  def __init__(self, path: str):
    self._examples = self.load_datapoints(path)

  def spec(self) -> lit_types.Spec:
    return {
        'source': lit_types.CategoryLabel(),
        'prompt': lit_types.TextSegment(),
        'target': lit_types.TextSegment(),
    }

  def load(self, path: str):
    return lit_dataset.Dataset(base=self, examples=self.load_datapoints(path))


class BillionWordBenchmark(lit_dataset.Dataset):
  """Billion Word Benchmark (lm1b); see http://www.statmt.org/lm-benchmark/."""

  AVAILABLE_SPLITS = ['test', 'train']

  def __init__(self, split: str = 'train', max_examples: Optional[int] = None):
    ds = tfds.load('lm1b', split=split)
    if max_examples is not None:
      # Normally we can just slice the resulting dataset, but lm1b is very large
      # so we can use ds.take() to only load a portion of it.
      ds = ds.take(max_examples)
    raw_examples = list(tfds.as_numpy(ds))
    self._examples = [{
        'text': ex['text'].decode('utf-8')
    } for ex in raw_examples]

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        'split': lit_types.CategoryLabel(vocab=cls.AVAILABLE_SPLITS),
        'max_examples': lit_types.Integer(
            default=1000, min_val=0, max_val=10_000, required=False
        ),
    }

  def spec(self) -> lit_types.Spec:
    return {'text': lit_types.TextSegment()}
