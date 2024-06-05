"""Text classification datasets, including single- and two-sentence tasks."""
from typing import Optional

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import tensorflow_datasets as tfds


def load_tfds(*args, **kw):
  """Load from TFDS."""
  # Materialize to NumPy arrays.
  # This also ensures compatibility with TF1.x non-eager mode, which doesn't
  # support direct iteration over a tf.data.Dataset.
  return list(
      tfds.as_numpy(tfds.load(*args, download=True, try_gcs=True, **kw)))


class IMDBData(lit_dataset.Dataset):
  """IMDB reviews dataset; see http://ai.stanford.edu/~amaas/data/sentiment/."""

  LABELS = ["0", "1"]
  AVAILABLE_SPLITS = ["test", "train", "unsupervised"]

  def __init__(
      self, split="test", max_seq_len=500, max_examples: Optional[int] = None
  ):
    """Dataset constructor, loads the data into memory."""
    raw_examples = load_tfds("imdb_reviews", split=split)
    self._examples = []  # populate this with data records
    for record in raw_examples[:max_examples]:
      # format and truncate from the end to max_seq_len tokens.
      truncated_text = " ".join(
          record["text"]
          .decode("utf-8")
          .replace("<br />", "")
          .split()[-max_seq_len:]
      )
      self._examples.append({
          "text": truncated_text,
          "label": self.LABELS[record["label"]],
      })

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        "split": lit_types.CategoryLabel(vocab=cls.AVAILABLE_SPLITS),
        "max_seq_len": lit_types.Integer(default=500, min_val=1, max_val=1024),
        "max_examples": lit_types.Integer(
            default=1000, min_val=0, max_val=10_000, required=False
        ),
    }

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "text": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=self.LABELS),
    }

