"""Text classification dataset for binary, single input data."""
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import pandas as pd


class SingleInputClassificationFromTSV(lit_dataset.Dataset):
  """TSV data loader for files having a single input text and a label.

  Files must be in TSV format with 2 columns in this order:
  1. Input text.
  2. Numeric label.

  Exported examples have 2 output keys: "sentence" and "label".
  """

  LABELS = ["0", "1"]

  def __init__(self, path: str):
    self._examples = self.load_datapoints(path)

  def load_datapoints(self, path: str):
    with open(path) as fd:
      df = pd.read_csv(fd, sep="\t", header=None, names=["sentence", "label"])
    # pylint: disable=g-complex-comprehension
    return [{
        "sentence": row["sentence"],
        "label": self.LABELS[row["label"]],
    } for _, row in df.iterrows()]
    # pylint: enable=g-complex-comprehension

  def spec(self) -> lit_types.Spec:
    return {
        "sentence": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=self.LABELS),
    }
