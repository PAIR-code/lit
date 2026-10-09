"""GLUE benchmark datasets, from the HuggingFace datasets hub.

See https://gluebenchmark.com/ and
https://huggingface.co/datasets/glue

The resulting LIT datasets contain regular Python/NumPy data.
"""
from typing import Optional

from datasets import load_dataset
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.lib import file_cache
from lit_nlp.lib import utils
import pandas as pd


def load_hf_glue(config_name: str, split: str, **kw):
  """Load a GLUE config from HuggingFace, preserving original example order."""
  return list(load_dataset('glue', config_name, split=split, **kw))


class CoLAData(lit_dataset.Dataset):
  """Corpus of Linguistic Acceptability.

  See https://huggingface.co/datasets/glue/viewer/cola.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_hf_glue('cola', split=split):
      self._examples.append({
          'sentence': ex['sentence'],
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'sentence': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class SST2Data(lit_dataset.Dataset):
  """Stanford Sentiment Treebank, binary version (SST-2).

  See https://huggingface.co/datasets/glue/viewer/sst2.
  """

  LABELS = ['0', '1']
  SPLITS = ['test', 'train', 'validation']

  def load_from_csv(self, path: str):
    path = file_cache.cached_path(path)
    with open(path) as fd:
      df = pd.read_csv(fd)
    if set(df.columns) != set(self.spec().keys()):
      raise ValueError(
          f'CSV columns {list(df.columns)} do not match expected'
          f' {list(self.spec().keys())}.'
      )
    df['label'] = df.label.map(str)
    return df.to_dict(orient='records')

  def load_from_hf(self, split: str):
    if split not in self.SPLITS:
      raise ValueError(
          f"Unsupported split '{split}'. Allowed values: {self.SPLITS}"
      )
    ret = []
    for ex in load_hf_glue('sst2', split=split):
      ret.append({
          'sentence': ex['sentence'],
          'label': self.LABELS[ex['label']],
      })
    return ret

  def __init__(
      self, path_or_splitname: str, max_examples: Optional[int] = None
  ):
    if path_or_splitname.endswith('.csv'):
      self._examples = self.load_from_csv(path_or_splitname)[:max_examples]
    else:
      self._examples = self.load_from_hf(path_or_splitname)[:max_examples]

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        'path_or_splitname': lit_types.String(
            default='validation', required=True
        ),
        'max_examples': lit_types.Integer(
            default=1000, min_val=0, max_val=10_000, required=False
        ),
    }

  def spec(self):
    return {
        'sentence': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class SST2DataForLM(SST2Data):
  """Stanford Sentiment Treebank, binary version (SST-2).

  See https://huggingface.co/datasets/glue/viewer/sst2.
  This data is reformatted to serve the language models.
  """

  def __init__(self, path_or_splitname: str, max_examples: int = -1):
    super().__init__(path_or_splitname, max_examples)
    self._examples = [
        utils.remap_dict(ex, {'sentence': 'text'}) for ex in self._examples
    ]

  def spec(self):
    return {
        'text': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS),
    }

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        'path_or_splitname': lit_types.String(
            default='validation', required=True
        ),
        'max_examples': lit_types.Integer(
            default=1000, min_val=0, max_val=10_000, required=False
        ),
    }


class MRPCData(lit_dataset.Dataset):
  """Microsoft Research Paraphrase Corpus.

  See https://huggingface.co/datasets/glue/viewer/mrpc.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_hf_glue('mrpc', split=split):
      self._examples.append({
          'sentence1': ex['sentence1'],
          'sentence2': ex['sentence2'],
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'sentence1': lit_types.TextSegment(),
        'sentence2': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class QQPData(lit_dataset.Dataset):
  """Quora Question Pairs.

  See https://huggingface.co/datasets/glue/viewer/qqp.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_hf_glue('qqp', split=split):
      self._examples.append({
          'question1': ex['question1'],
          'question2': ex['question2'],
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'question1': lit_types.TextSegment(),
        'question2': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class STSBData(lit_dataset.Dataset):
  """Semantic Textual Similarity Benchmark (STS-B).

  Unlike the other GLUE tasks, this is formulated as a regression problem.

  See https://huggingface.co/datasets/glue/viewer/stsb.
  """
  SPLITS = ['test', 'train', 'validation']

  def load_from_csv(self, path: str):
    path = file_cache.cached_path(path)
    with open(path) as fd:
      df = pd.read_csv(fd)
    if set(df.columns) != set(self.spec().keys()):
      raise ValueError(
          f'CSV columns {list(df.columns)} do not match expected'
          f' {list(self.spec().keys())}.'
      )
    df['label'] = df.label.map(float)
    return df.to_dict(orient='records')

  def load_from_hf(self, split: str):
    if split not in self.SPLITS:
      raise ValueError(
          f"Unsupported split '{split}'. Allowed values: {self.SPLITS}"
      )
    ret = []
    for ex in load_hf_glue('stsb', split=split):
      ret.append({
          'sentence1': ex['sentence1'],
          'sentence2': ex['sentence2'],
          'label': ex['label'],
      })
    return ret

  def __init__(
      self, path_or_splitname: str, max_examples: Optional[int] = None
  ):
    if path_or_splitname.endswith('.csv'):
      self._examples = self.load_from_csv(path_or_splitname)[:max_examples]
    else:
      self._examples = self.load_from_hf(path_or_splitname)[:max_examples]

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        'path_or_splitname': lit_types.String(
            default='validation', required=True
        ),
        'max_examples': lit_types.Integer(
            default=1000, min_val=0, max_val=10_000, required=False
        ),
    }

  def spec(self):
    return {
        'sentence1': lit_types.TextSegment(),
        'sentence2': lit_types.TextSegment(),
        'label': lit_types.Scalar(min_val=0, max_val=5),
    }


class MNLIData(lit_dataset.Dataset):
  """MultiNLI dataset.

  See https://huggingface.co/datasets/glue/viewer/mnli.
  """

  LABELS = ['entailment', 'neutral', 'contradiction']
  SPLITS = [
      'test_matched',
      'test_mismatched',
      'train',
      'validation_matched',
      'validation_mismatched',
  ]

  def load_from_csv(self, path: str):
    path = file_cache.cached_path(path)
    with open(path) as fd:
      df = pd.read_csv(fd)
    if set(df.columns) != set(self.spec().keys()):
      raise ValueError(
          f'CSV columns {list(df.columns)} do not match expected'
          f' {list(self.spec().keys())}.'
      )
    df['label'] = df.label.map(str)
    return df.to_dict(orient='records')

  def load_from_hf(self, split: str):
    if split not in self.SPLITS:
      raise ValueError(
          f"Unsupported split '{split}'. Allowed values: {self.SPLITS}"
      )
    ret = []
    for ex in load_hf_glue('mnli', split=split):
      ret.append({
          'premise': ex['premise'],
          'hypothesis': ex['hypothesis'],
          'label': self.LABELS[ex['label']],
      })
    return ret

  def __init__(
      self, path_or_splitname: str, max_examples: Optional[int] = None
  ):
    if path_or_splitname.endswith('.csv'):
      self._examples = self.load_from_csv(path_or_splitname)[:max_examples]
    else:
      self._examples = self.load_from_hf(path_or_splitname)[:max_examples]

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        'path_or_splitname': lit_types.String(
            default='validation_matched', required=True
        ),
        'max_examples': lit_types.Integer(
            default=1000, min_val=0, max_val=10_000, required=False
        ),
    }

  def spec(self):
    return {
        'premise': lit_types.TextSegment(),
        'hypothesis': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class QNLIData(lit_dataset.Dataset):
  """NLI examples derived from SQuAD.

  See https://huggingface.co/datasets/glue/viewer/qnli.
  """

  LABELS = ['entailment', 'not_entailment']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_hf_glue('qnli', split=split):
      self._examples.append({
          'question': ex['question'],
          'sentence': ex['sentence'],
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'question': lit_types.TextSegment(),
        'sentence': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class RTEData(lit_dataset.Dataset):
  """Recognizing Textual Entailment.

  See https://huggingface.co/datasets/glue/viewer/rte.
  """

  LABELS = ['entailment', 'not_entailment']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_hf_glue('rte', split=split):
      self._examples.append({
          'sentence1': ex['sentence1'],
          'sentence2': ex['sentence2'],
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'sentence1': lit_types.TextSegment(),
        'sentence2': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class WNLIData(lit_dataset.Dataset):
  """Winograd schema challenge.

  See https://huggingface.co/datasets/glue/viewer/wnli.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_hf_glue('wnli', split=split):
      self._examples.append({
          'sentence1': ex['sentence1'],
          'sentence2': ex['sentence2'],
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'sentence1': lit_types.TextSegment(),
        'sentence2': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class DiagnosticNLIData(lit_dataset.Dataset):
  """NLI diagnostic set; use to evaluate models trained on MultiNLI.

  See https://huggingface.co/datasets/glue/viewer/ax.
  """

  LABELS = ['entailment', 'neutral', 'contradiction']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_hf_glue('ax', split=split):
      self._examples.append({
          'premise': ex['premise'],
          'hypothesis': ex['hypothesis'],
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'premise': lit_types.TextSegment(),
        'hypothesis': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }
