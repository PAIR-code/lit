"""Methods for configuring prompt debugging datasets."""

from collections.abc import Mapping, Sequence
import copy
import functools
import json
import os
import re
from typing import Optional

from absl import logging
from lit_nlp import app as lit_app
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

SAMPLE_DATA_DIR = os.path.dirname(__file__)
DEFAULT_DATASETS = ['sample_prompts']
DEFAULT_MAX_EXAMPLES = 1000


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
        'path_or_glob': lit_types.String(default=default_path, required=False),
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


_plaintext_prompts = functools.partial(  # pylint: disable=invalid-name
    PlaintextSents, field_name='prompt'
)
# Hack: normally dataset loaders are a class object which has a __name__,
# rather than a functools.partial
_plaintext_prompts.__name__ = 'PlaintextSents'


def get_datasets(
    datasets_config: Optional[Sequence[str]] = None,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> Mapping[str, lit_dataset.Dataset]:
  """Loads datasets from the provided configs.

  Args:
    datasets_config: A sequence of configs in the form of <name>:<path> where
      the path points to is either: 1) a JSON Lines file containing records with
      a required "prompt" field and optional "target" and "source" fields; or 2)
      a plain text file where each line is a prompt.
    max_examples: Maximum number of examples in each loaded dataset.

  Returns:
    A mapping from dataset name to the initialized LIT dataset.
  """

  if not datasets_config:
    return {}

  datasets: dict[str, lit_dataset.Dataset] = {}
  for dataset_string in datasets_config:
    if dataset_string == 'sample_prompts':
      dataset_name = 'sample_prompts'
      path = PromptExamples.SAMPLE_DATA_PATH
    else:
      # Only split on the first ':', because path may be a URL
      # containing 'https://'
      dataset_name, path = dataset_string.split(':', 1)
    logging.info("Loading dataset '%s' from '%s'", dataset_name, path)

    if path.endswith('.jsonl'):
      datasets[dataset_name] = PromptExamples(path)
    # .txt or .txt-#####-of-#####
    elif path.endswith('.txt') or re.match(r'.*\.txt-\d{5}-of-\d{5}$', path):
      datasets[dataset_name] = _plaintext_prompts(path)
    else:
      raise ValueError(f'Unsupported dataset format for {dataset_string}')

  for name in datasets:
    datasets[name] = datasets[name].slice[:max_examples]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))

  return datasets


def get_dataset_loaders() -> lit_app.DatasetLoadersMap:
  return {
      'jsonl_examples': (
          PromptExamples,
          PromptExamples.init_spec(),
      ),
      'plaintext_inputs': (
          _plaintext_prompts,
          PlaintextSents.init_spec(),
      ),
  }
