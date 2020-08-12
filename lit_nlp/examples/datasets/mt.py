# Lint as: python3
"""Machine translation datasets."""

from typing import List

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import six
import tensorflow_datasets as tfds

JsonDict = lit_types.JsonDict
Spec = lit_types.Spec


class WMT14Data(lit_dataset.Dataset):
  """WMT '14 machine-translation data, via TFDS."""

  def __init__(self, version='fr-en', reverse=False):
    self._builder = tfds.builder('wmt14_translate', config=version)

    lang_keys = self._builder.info.supervised_keys
    if reverse:
      self.source_key = lang_keys[1]
      self.target_key = lang_keys[0]
    else:
      self.source_key = lang_keys[0]
      self.target_key = lang_keys[1]

    # Pre-load dataset
    self._examples = self._get_examples()

  def spec(self) -> Spec:
    return {
        'source': lit_types.TextSegment(),
        'source_language': lit_types.CategoryLabel(),
        'target': lit_types.TextSegment(),
        'target_language': lit_types.CategoryLabel(),
    }

  def _get_examples(self) -> List[JsonDict]:
    """Get validation set for this language pair."""
    ds = self._builder.as_dataset(split='validation')
    # TODO(lit-team): don't load the whole dataset if only using a few examples
    ds_np = list(tfds.as_numpy(ds))
    # pylint: disable=g-complex-comprehension
    ret = [{
        'source': six.ensure_text(d[self.source_key]),
        'source_language': self.source_key,
        'target': six.ensure_text(d[self.target_key]),
        'target_language': self.target_key,
    } for d in ds_np]
    # pylint: enable=g-complex-comprehension
    return ret
