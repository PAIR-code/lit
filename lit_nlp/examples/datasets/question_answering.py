"""Data loaders for summarization datasets."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import tensorflow_datasets as tfds


class TyDiQA(lit_dataset.Dataset):
  """TyDiQA dataset."""

  def __init__(self, split="validation-en", max_examples=-1, max_seq_len=500):
    """Dataset constructor, loads the data into memory."""
    ds = tfds.load("tydi_qa", split=split)

    # into datafrane to decode string
    df = tfds.as_dataframe(ds.take(max_examples))
    df['context'] = df['context'].str.decode("utf-8")
    df['question'] = df['question'].str.decode("utf-8")

    # populate this with data records
    self._examples = [{
      'context': row['context'],
      'question': row['question'],
    } for _, row in df.iterrows()]

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "context": lit_types.TextSegment(),
        "question": lit_types.TextSegment(),
    }