"""Data loaders for summarization datasets."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import tensorflow_datasets as tfds


class GigawordData(lit_dataset.Dataset):
  """English Gigaword summarization dataset."""

  def __init__(self, split="validation", max_examples=-1):
    """Dataset constructor, loads the data into memory."""
    ds = tfds.load("gigaword", split=split)

    self._examples = []  # populate this with data records
    for record in ds.take(max_examples):
      input_text = record["document"].numpy().decode("utf-8")
      output_text = record["summary"].numpy().decode("utf-8")
      self._examples.append({
          "input_text": input_text,
          "output_text": output_text,
      })

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "input_text": lit_types.TextSegment(),
        "output_text": lit_types.TextSegment(),
    }


class CNNDMData(lit_dataset.Dataset):
  """English CNNDM summarization dataset."""

  def __init__(self, split="validation", max_examples=-1, max_seq_len=500):
    """Dataset constructor, loads the data into memory."""
    ds = tfds.load("cnn_dailymail", split=split)

    self._examples = []  # populate this with data records
    for record in ds.take(max_examples):
      # format and truncate from the end to max_seq_len tokens.
      input_text = " ".join(
          record["article"].numpy()\
                        .decode("utf-8")\
                        .replace("<br />", "")\
                        .split()[-max_seq_len:])
      output_text = record["highlights"].numpy().decode("utf-8")
      self._examples.append({
          "input_text": input_text,
          "output_text": output_text,
      })

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "input_text": lit_types.TextSegment(),
        "output_text": lit_types.TextSegment(),
    }
