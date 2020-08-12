# Lint as: python3
"""Text classification datasets, including single- and two-sentence tasks."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import pandas as pd
import tensorflow_datasets as tfds




class MNLIDataFromTSV(lit_dataset.Dataset):
  """MultiNLI dataset, from TSV.

  Compared to the TFDS version, this includes:
  - label2 field for binary labels, with same schema as HANS
  - genre labels, for stratified analysis

  The downside is that you need to download the data from
  https://gluebenchmark.com/tasks, and provide a path to the .tsv file.
  """

  LABELS3 = ["contradiction", "entailment", "neutral"]
  LABELS2 = ["non-entailment", "entailment"]

  def binarize_label(self, label):
    return "entailment" if label == "entailment" else "non-entailment"

  def __init__(self, path: str):
    with open(path) as fd:
      df = pd.read_csv(fd, sep="\t")
    # pylint: disable=g-complex-comprehension
    self._examples = [{
        "premise": row["sentence1"],
        "hypothesis": row["sentence2"],
        "label": row["gold_label"],
        "label2": self.binarize_label(row["gold_label"]),
        "genre": row["genre"],
    } for _, row in df.iterrows()]
    # pylint: enable=g-complex-comprehension

  def spec(self) -> lit_types.Spec:
    """Should match MnliModel's input_spec()."""
    return {
        "premise": lit_types.TextSegment(),
        "hypothesis": lit_types.TextSegment(),
        # 'label' for 3-way NLI labels, 'label2' for binarized.
        "label": lit_types.CategoryLabel(vocab=self.LABELS3),
        "label2": lit_types.CategoryLabel(vocab=self.LABELS2),
        "genre": lit_types.CategoryLabel(),
    }


class HansNLIData(lit_dataset.Dataset):
  """HANS NLI challenge set (https://arxiv.org/abs/1902.01007); 30k examples."""

  LABELS = ["non-entailment", "entailment"]

  def __init__(self, path: str):
    with open(path) as fd:
      df = pd.read_csv(fd, sep="\t", header=0)
    # pylint: disable=g-complex-comprehension
    self._examples = [{
        "premise": row["sentence1"],
        "hypothesis": row["sentence2"],
        "label2": row["gold_label"],
        "heuristic": row["heuristic"],
        "template": row["template"],
    } for _, row in df.iterrows()]
    # pylint: enable=g-complex-comprehension

  def spec(self) -> lit_types.Spec:
    return {
        "premise": lit_types.TextSegment(),
        "hypothesis": lit_types.TextSegment(),
        # 'label2' for 2-way NLI labels
        "label2": lit_types.CategoryLabel(vocab=self.LABELS),
        "heuristic": lit_types.CategoryLabel(),
        "template": lit_types.CategoryLabel(),
    }


class IMDBData(lit_dataset.Dataset):
  """IMDB reviews dataset; see http://ai.stanford.edu/~amaas/data/sentiment/."""

  LABELS = ["0", "1"]

  def __init__(self, split="test", max_seq_len=500):
    """Dataset constructor, loads the data into memory."""
    ds = tfds.load("imdb_reviews", split=split, download=True, try_gcs=True)
    raw_examples = list(tfds.as_numpy(ds))
    self._examples = []  # populate this with data records
    for record in raw_examples:
      # format and truncate from the end to max_seq_len tokens.
      truncated_text = " ".join(
          record["text"].decode("utf-8")\
                        .replace("<br />", "")\
                        .split()[-max_seq_len:])
      self._examples.append({
          "text": truncated_text,
          "label": self.LABELS[record["label"]],
      })

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "text": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=self.LABELS),
    }
