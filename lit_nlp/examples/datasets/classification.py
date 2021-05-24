# Lint as: python3
"""Text classification datasets, including single- and two-sentence tasks."""
from typing import List

from absl import logging
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import pandas as pd
import tensorflow_datasets as tfds


def load_tfds(*args, **kw):
  """Load from TFDS."""
  # Materialize to NumPy arrays.
  # This also ensures compatibility with TF1.x non-eager mode, which doesn't
  # support direct iteration over a tf.data.Dataset.
  return list(
      tfds.as_numpy(tfds.load(*args, download=True, try_gcs=True, **kw)))


class MNLIDataFromTSV(lit_dataset.Dataset):
  """MultiNLI dataset, from TSV.

  Compared to the TFDS version, this includes:
  - label2 field for binary labels, with same schema as HANS
  - genre labels, for stratified analysis

  The downside is that you need to download the data from
  https://gluebenchmark.com/tasks, and provide a path to the .tsv file.
  """

  LABELS3 = ["entailment", "neutral", "contradiction"]
  LABELS2 = ["non-entailment", "entailment"]

  def binarize_label(self, label):
    return "entailment" if label == "entailment" else "non-entailment"

  def __init__(self, path: str):
    self._examples = self.load_datapoints(path)

  def load_datapoints(self, path: str):
    with open(path) as fd:
      df = pd.read_csv(fd, sep="\t")
    # pylint: disable=g-complex-comprehension
    return [{
        "premise": row["sentence1"],
        "hypothesis": row["sentence2"],
        "label": row["gold_label"],
        "label2": self.binarize_label(row["gold_label"]),
        "genre": row["genre"],
    } for _, row in df.iterrows()]
    # pylint: enable=g-complex-comprehension

  def load(self, path: str):
    datapoints = self.load_datapoints(path)
    return lit_dataset.Dataset(base=self, examples=datapoints)

  def save(self, examples: List[lit_types.IndexedInput], path: str):
    example_data = [ex["data"] for ex in examples]
    df = pd.DataFrame(example_data).rename(columns={
        "premise": "sentence1",
        "hypothesis": "sentence2",
        "label": "gold_label",
    })
    with open(path, "w") as fd:
      df.to_csv(fd, sep="\t")

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


class XNLIData(lit_dataset.Dataset):
  """Cross-lingual NLI; see https://cims.nyu.edu/~sbowman/xnli/."""

  LABELS = ["entailment", "neutral", "contradiction"]

  def _process_example(self, ex, languages: List[str]):
    # Hypothesis is stored as parallel arrays, so make a map.
    hyp_map = {
        lang.decode("utf-8"): hyp.decode("utf-8") for lang, hyp in zip(
            ex["hypothesis"]["language"], ex["hypothesis"]["translation"])
    }
    for lang in languages:
      if lang not in hyp_map:
        logging.warning("Missing hypothesis (lang=%s) for premise '%s'", lang,
                        ex["premise"]["lang"].decode("utf-8"))
        continue
      yield {
          "premise": ex["premise"][lang].decode("utf-8"),
          "hypothesis": hyp_map[lang],
          "label": self.LABELS[ex["label"]],
          "language": lang,
      }

  def __init__(self, split: str, languages=("en", "es", "hi", "zh")):
    self._examples = []
    for ex in load_tfds("xnli", split=split):
      # Each TFDS example contains all the translations; we unpack to individual
      # (premise, hypothesis) pairs that are compatible with a standard NLI
      # model.
      self._examples.extend(self._process_example(ex, languages))

  def spec(self):
    return {
        "premise": lit_types.TextSegment(),
        "hypothesis": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=self.LABELS),
        "language": lit_types.CategoryLabel(),
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
    raw_examples = load_tfds("imdb_reviews", split=split)
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


class ToxicityData(lit_dataset.Dataset):
  """Jigsaw toxicity dataset; see https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes."""

  LABELS = ["non-toxic", "toxic"]

  def __init__(self, split="test", max_seq_len=500):
    """Dataset constructor, loads the data into memory."""
    raw_examples = load_tfds("wikipedia_toxicity_subtypes", split=split)
    self._examples = []  # populate this with data records
    for record in raw_examples:
      self._examples.append({
          "sentence": record["text"].decode("utf-8"),
          "label": self.LABELS[int(record["toxicity"])],
          "identity_attack": bool(int(record["identity_attack"])),
          "insult": bool(int(record["insult"])),
          "obscene": bool(int(record["obscene"])),
          "severe_toxicity": bool(int(record["severe_toxicity"])),
          "threat": bool(int(record["threat"]))
      })

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "sentence": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=self.LABELS),
        "identity_attack": lit_types.Boolean(),
        "insult": lit_types.Boolean(),
        "obscene": lit_types.Boolean(),
        "severe_toxicity": lit_types.Boolean(),
        "threat": lit_types.Boolean()
    }
