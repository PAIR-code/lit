"""Data loaders for summarization datasets."""

from typing import Optional
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import pandas as pd
import tensorflow_datasets as tfds


class GigawordData(lit_dataset.Dataset):
  """English Gigaword summarization dataset."""

  def __init__(self, split="validation", max_examples=-1):
    """Dataset constructor, loads the data into memory."""
    ds = tfds.load("gigaword", split=split)

    self._examples = []  # populate this with data records
    for record in ds.take(max_examples):
      self._examples.append({
          "document": record["document"].numpy().decode("utf-8"),
          "reference": record["summary"].numpy().decode("utf-8"),
      })

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "document": lit_types.TextSegment(),
        "reference": lit_types.TextSegment(),
    }


class CNNDMData(lit_dataset.Dataset):
  """English CNNDM summarization dataset."""

  tfds_name = "cnn_dailymail"

  def __init__(self,
               split: str = "validation",
               max_examples: int = -1,
               max_seq_len: int = 500,
               filepath: Optional[str] = None):
    """Initializes a Dataset wrapper for the the CNN DailyMail dataset.

    Args:
      split: The subset of the data to dowload from TFDS.
      max_examples: The number of examples to include from the TFDS dataset.
      max_seq_len: The maximum length to load for any document in the dataset.
      filepath: If provided, the dataset will be loaded from disk instead of
        downloaded from TFDS. This is useful for containerized demos where the
        process of downloading and processing the dataset with TFDS can cause
        the container to timeout and restart on a pseudo-inifinite loop.
    """

    if filepath:
      examples = self.load_datapoints(filepath)
    else:
      ds = tfds.load(self.tfds_name, split=split)
      examples = []
      for record in ds.take(max_examples):
        # format and truncate from the end to max_seq_len tokens.
        document = " ".join(record["article"].numpy().decode("utf-8").replace(
            "<br />", "").split()[-max_seq_len:])
        reference = record["highlights"].numpy().decode("utf-8")
        examples.append({"document": document, "reference": reference})

    self._examples = examples

  def load_datapoints(self,
                      path: str,
                      max_seq_len: int = 500,
                      skiplines: int = 0) -> list[lit_types.JsonDict]:
    """Loads the dataset from a CSV file.

    Args:
      path: The location of the CSV file.
      max_seq_len: The maximum length to load for any document in the dataset.
      skiplines: The number of lines to skip in the input file(s).

    Returns:
      The list of exmaples loaded from the file.
    """
    examples = []
    df = pd.read_csv(path, skiprows=skiplines)
    for _, row in df.iterrows():
      examples.append({
          "document": row["article"][-max_seq_len:],
          "reference": row["highlights"],
      })
    return examples

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "document": lit_types.TextSegment(),
        "reference": lit_types.TextSegment(),
    }
