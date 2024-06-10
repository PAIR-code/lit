"""GLUE benchmark datasets, using TFDS or from CSV.

See https://gluebenchmark.com/ and
https://www.tensorflow.org/datasets/catalog/glue

Note that this requires the TensorFlow Datasets package, but the resulting LIT
datasets just contain regular Python/NumPy data.
"""
from typing import Optional

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.lib import file_cache
from lit_nlp.lib import utils
import pandas as pd
import tensorflow_datasets as tfds


def load_tfds(*args, do_sort=True, **kw):
  """Load from TFDS, with optional sorting."""
  # Materialize to NumPy arrays.
  # This also ensures compatibility with TF1.x non-eager mode, which doesn't
  # support direct iteration over a tf.data.Dataset.
  ret = list(tfds.as_numpy(tfds.load(*args, download=True, try_gcs=True, **kw)))
  if do_sort:
    # Recover original order, as if you loaded from a TSV file.
    ret.sort(key=lambda ex: ex['idx'])
  return ret


class CoLAData(lit_dataset.Dataset):
  """Corpus of Linguistic Acceptability.

  See
  https://www.tensorflow.org/datasets/catalog/glue#gluecola_default_config.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_tfds('glue/cola', split=split):
      self._examples.append({
          'sentence': ex['sentence'].decode('utf-8'),
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'sentence': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }


class SST2Data(lit_dataset.Dataset):
  """Stanford Sentiment Treebank, binary version (SST-2).

  See https://www.tensorflow.org/datasets/catalog/glue#gluesst2.
  """

  LABELS = ['0', '1']
  TFDS_SPLITS = ['test', 'train', 'validation']

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

  def load_from_tfds(self, split: str):
    if split not in self.TFDS_SPLITS:
      raise ValueError(
          f"Unsupported split '{split}'. Allowed values: {self.TFDS_SPLITS}"
      )
    ret = []
    for ex in load_tfds('glue/sst2', split=split):
      ret.append({
          'sentence': ex['sentence'].decode('utf-8'),
          'label': self.LABELS[ex['label']],
      })
    return ret

  def __init__(
      self, path_or_splitname: str, max_examples: Optional[int] = None
  ):
    if path_or_splitname.endswith('.csv'):
      self._examples = self.load_from_csv(path_or_splitname)[:max_examples]
    else:
      self._examples = self.load_from_tfds(path_or_splitname)[:max_examples]

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

  See https://www.tensorflow.org/datasets/catalog/glue#gluesst2.
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

  See https://www.tensorflow.org/datasets/catalog/glue#gluemrpc.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_tfds('glue/mrpc', split=split):
      self._examples.append({
          'sentence1': ex['sentence1'].decode('utf-8'),
          'sentence2': ex['sentence2'].decode('utf-8'),
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

  See https://www.tensorflow.org/datasets/catalog/glue#glueqqp.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_tfds('glue/qqp', split=split):
      self._examples.append({
          'question1': ex['question1'].decode('utf-8'),
          'question2': ex['question2'].decode('utf-8'),
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

  See https://www.tensorflow.org/datasets/catalog/glue#gluestsb.
  """
  TFDS_SPLITS = ['test', 'train', 'validation']

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

  def load_from_tfds(self, split: str):
    if split not in self.TFDS_SPLITS:
      raise ValueError(
          f"Unsupported split '{split}'. Allowed values: {self.TFDS_SPLITS}"
      )
    ret = []
    for ex in load_tfds('glue/stsb', split=split):
      ret.append({
          'sentence1': ex['sentence1'].decode('utf-8'),
          'sentence2': ex['sentence2'].decode('utf-8'),
          'label': ex['label'],
      })
    return ret

  def __init__(
      self, path_or_splitname: str, max_examples: Optional[int] = None
  ):
    if path_or_splitname.endswith('.csv'):
      self._examples = self.load_from_csv(path_or_splitname)[:max_examples]
    else:
      self._examples = self.load_from_tfds(path_or_splitname)[:max_examples]

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

  See https://www.tensorflow.org/datasets/catalog/glue#gluemnli.
  """

  LABELS = ['entailment', 'neutral', 'contradiction']
  TFDS_SPLITS = [
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

  def load_from_tfds(self, split: str):
    if split not in self.TFDS_SPLITS:
      raise ValueError(
          f"Unsupported split '{split}'. Allowed values: {self.TFDS_SPLITS}"
      )
    ret = []
    for ex in load_tfds('glue/mnli', split=split):
      ret.append({
          'premise': ex['premise'].decode('utf-8'),
          'hypothesis': ex['hypothesis'].decode('utf-8'),
          'label': self.LABELS[ex['label']],
      })
    return ret

  def __init__(
      self, path_or_splitname: str, max_examples: Optional[int] = None
  ):
    if path_or_splitname.endswith('.csv'):
      self._examples = self.load_from_csv(path_or_splitname)[:max_examples]
    else:
      self._examples = self.load_from_tfds(path_or_splitname)[:max_examples]

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

  See https://www.tensorflow.org/datasets/catalog/glue#glueqnli.
  """

  LABELS = ['entailment', 'not_entailment']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_tfds('glue/qnli', split=split):
      self._examples.append({
          'question': ex['question'].decode('utf-8'),
          'sentence': ex['sentence'].decode('utf-8'),
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

  See https://www.tensorflow.org/datasets/catalog/glue#gluerte.
  """

  LABELS = ['entailment', 'not_entailment']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_tfds('glue/rte', split=split):
      self._examples.append({
          'sentence1': ex['sentence1'].decode('utf-8'),
          'sentence2': ex['sentence2'].decode('utf-8'),
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

  See https://www.tensorflow.org/datasets/catalog/glue#gluewnli.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_tfds('glue/wnli', split=split):
      self._examples.append({
          'sentence1': ex['sentence1'].decode('utf-8'),
          'sentence2': ex['sentence2'].decode('utf-8'),
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

  See https://www.tensorflow.org/datasets/catalog/glue#glueax.
  """

  LABELS = ['entailment', 'neutral', 'contradiction']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_tfds('glue/ax', split=split):
      self._examples.append({
          'premise': ex['premise'].decode('utf-8'),
          'hypothesis': ex['hypothesis'].decode('utf-8'),
          'label': self.LABELS[ex['label']],
      })

  def spec(self):
    return {
        'premise': lit_types.TextSegment(),
        'hypothesis': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }
