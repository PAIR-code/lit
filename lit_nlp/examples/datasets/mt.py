"""Machine translation datasets."""

from typing import Any, Mapping, Optional

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import pandas as pd
import six
import tensorflow_datasets as tfds

JsonDict = lit_types.JsonDict
Spec = lit_types.Spec

_VALID_WMT14_LANG_CODES = ('cs', 'de', 'fr', 'en', 'hi', 'ru')
_VALID_WMT14_LANG_PAIRS = ('cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en')


class WMT14Data(lit_dataset.Dataset):
  """WMT '14 machine-translation data, via TFDS."""

  tfds_name = 'wmt14_translate'

  def __init__(self,
               version: str = 'fr-en',
               reverse: bool = False,
               split: str = 'validation',
               filepath: Optional[str] = None):
    """Initializes a Dataset wrapper for the WMT-14 dataset.

    More information on the WMT 14 dataset can be found at the links below.

    * [WMT 14 website](https://www.statmt.org/wmt14/translation-task.html)
    * [TFDS docs](https://www.tensorflow.org/datasets/catalog/wmt14_translate)
    * [Bojar et al. 2014](https://aclanthology.org/W14-3302/)

    Args:
      version: An identifier for the dataset comprising two different [ISO
        639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) language
        codes separated by a `-`. WMT14 supports a limited subset of languages,
        valid versions include `cs-en`, `de-en`, `fr-en`, `hi-en`, and `ru-en`.
      reverse: If true, invert the source and target languages.
      split: The subset of the data to dowload from TFDS.
      filepath: If provided, the dataset will be loaded from disk instead of
        downloaded from TFDS. This is useful for containerized demos where the
        process of downloading and processing the dataset with TFDS can cause
        the container to timeout and restart on a pseudo-inifinite loop.

    Raises:
      ValueError: Invalid version identifier.
    """
    if version not in _VALID_WMT14_LANG_PAIRS:
      raise ValueError(f'Invalid version, {version}. Expected one of '
                       f'{str(_VALID_WMT14_LANG_PAIRS)}.')

    lang_keys = version.split('-')
    if reverse:
      source_key = lang_keys[1]
      target_key = lang_keys[0]
    else:
      source_key = lang_keys[0]
      target_key = lang_keys[1]

    # Load dataset
    if filepath:
      self._examples = self._load_datapoints(filepath, source_key, target_key)
    else:
      ds_name = f'{self.tfds_name}/{version}'
      ds = tfds.load(ds_name, download=True, try_gcs=True, split=split)
      ds_np = list(tfds.as_numpy(ds))
      self._examples = [
          self._record_to_dict(d, source_key, target_key) for d in ds_np
      ]

  def _load_datapoints(self,
                       path: str,
                       source_key: str,
                       target_key: str,
                       skiplines: int = 0) -> list[JsonDict]:
    """Loads the dataset from a CSV file.

    Args:
      path: The location of the CSV file.
      source_key: ISO 639-1 code for the source language.
      target_key: ISO 639-1 code for the target language.
      skiplines: The number of lines to skip in the input file(s).

    Returns:
      The list of exmaples loaded from the file.

    Raises:
      ValueError: invalid key passed for the source or target language.
    """
    if source_key not in _VALID_WMT14_LANG_CODES:
      raise ValueError(f'Invalid source_key, {source_key}. Expected one of '
                       f'{str(_VALID_WMT14_LANG_CODES)}.')

    if target_key not in _VALID_WMT14_LANG_CODES:
      raise ValueError(f'Invalid target_key, {target_key}. Expected one of '
                       f'{str(_VALID_WMT14_LANG_CODES)}.')

    df = pd.read_csv(path, skiprows=skiplines)
    return [
        self._record_to_dict(row, source_key, target_key)
        for _, row in df.iterrows()
    ]

  def _record_to_dict(
      self,
      record: Mapping[str, Any],
      source_key: str,
      target_key: str,
  ) -> JsonDict:
    """Converts a record (typically np.array or pd.Series) into a JsonDict."""
    return {
        'source': six.ensure_text(record[source_key]),
        'source_language': source_key,
        'target': six.ensure_text(record[target_key]),
        'target_language': target_key,
    }

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
