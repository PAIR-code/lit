# Lint as: python3
"""Machine translation datasets."""

from typing import Optional

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import six
import tensorflow_datasets as tfds

JsonDict = lit_types.JsonDict
Spec = lit_types.Spec


class WMT14Data(lit_dataset.Dataset):
  """WMT '14 machine-translation data, via TFDS."""

  def __init__(self, version='fr-en', reverse=False, split: str = 'validation'):
    lang_keys = version.split('-')
    assert len(lang_keys) == 2
    if reverse:
      source_key = lang_keys[1]
      target_key = lang_keys[0]
    else:
      source_key = lang_keys[0]
      target_key = lang_keys[1]

    # Pre-load dataset
    ds_name = 'wmt14_translate/' + version
    ds = tfds.load(ds_name, download=True, try_gcs=True, split=split)
    # TODO(lit-team): don't load the whole dataset if only using a few examples
    ds_np = list(tfds.as_numpy(ds))
    # pylint: disable=g-complex-comprehension
    self._examples = [{
        'source': six.ensure_text(d[source_key]),
        'source_language': source_key,
        'target': six.ensure_text(d[target_key]),
        'target_language': target_key,
    } for d in ds_np]
    # pylint: enable=g-complex-comprehension

  def spec(self) -> Spec:
    return {
        'source': lit_types.TextSegment(),
        'source_language': lit_types.CategoryLabel(),
        'target': lit_types.TextSegment(),
        'target_language': lit_types.CategoryLabel(),
    }


class WMT17Data(lit_dataset.Dataset):
  """WMT '17 machine-translation data, via TFDS."""

  def __init__(self,
               version: str = 'de-en',
               reverse: bool = False,
               split: str = 'validation',
               max_examples: Optional[int] = None):
    lang_keys = version.split('-')
    assert len(lang_keys) == 2
    if reverse:
      source_key = lang_keys[1]
      target_key = lang_keys[0]
    else:
      source_key = lang_keys[0]
      target_key = lang_keys[1]

    # Pre-load dataset
    ds_name = 'wmt17_translate/' + version
    ds = tfds.load(ds_name, download=True, try_gcs=True, split=split)
    ds_np = list(tfds.as_numpy(ds))
    # pylint: disable=g-complex-comprehension
    self._examples = [{
        'source': six.ensure_text(d[source_key]),
        'source_language': source_key,
        'target': six.ensure_text(d[target_key]),
        'target_language': target_key,
    } for d in ds_np]
    # pylint: enable=g-complex-comprehension
    self._examples = self._examples[:max_examples]

  def spec(self) -> Spec:
    return {
        'source': lit_types.TextSegment(),
        'source_language': lit_types.CategoryLabel(required=False),
        'target': lit_types.TextSegment(),
        'target_language': lit_types.CategoryLabel(required=False),
    }
