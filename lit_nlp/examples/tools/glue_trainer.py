r"""Lightweight trainer script to fine-tune on a GLUE or GLUE-like task.

Usage:
  python -m lit_nlp.examples.tools.glue_trainer \
    --encoder_name=bert-base-uncased --task=sst2 \
    --train_path=/path/to/save/model

For a quick start, use:
   --encoder_name="google/bert_uncased_L-2_H-128_A-2"

This will train a "bert-tiny" model from https://arxiv.org/abs/1908.08962,
which should run in under five minutes on a single GPU, and give validation
accuracy in the low 80s on SST-2.

Note: you don't have to use this trainer to use LIT; the classifier
implementation is just a wrapper around HuggingFace Transformers, using
AutoTokenizer, AutoConfig, and AutoModelForSequenceClassification, and can
load anything compatible with those classes.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from lit_nlp.examples.glue import data as glue_data
from lit_nlp.examples.glue import models as glue_models
from lit_nlp.lib import serialize

_ENCODER_NAME = flags.DEFINE_string(
    "encoder_name", "bert-base-uncased",
    "Model name or path to pretrained (base) encoder.")
_TASK = flags.DEFINE_string("task", "sst2", "Name of task to fine-tune on.")
_TRAIN_PATH = flags.DEFINE_string("train_path", "/tmp/hf_demo",
                                  "Path to save fine-tuned model.")

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 3, "Number of epochs to train for.", lower_bound=1)
_SAVE_INTERMEDIATES = flags.DEFINE_bool(
    "save_intermediates", False,
    "If true, save intermediate weights after each epoch.")

FLAGS = flags.FLAGS


def train_and_save(model,
                   train_data,
                   val_data,
                   train_path,
                   save_intermediates=False,
                   **train_kw):
  """Run training and save model."""

  def epoch_saver(epoch, logs):
    del logs  # unused
    if not save_intermediates:
      return
    # Save path 1-indexed = # of completed epochs.
    save_path = os.path.join(train_path, f"epoch-{epoch+1}")
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)

  history = model.train(
      train_data.examples,
      validation_inputs=val_data.examples,
      on_epoch_end=epoch_saver,
      **train_kw)

  # Save training history too, since this is human-readable and more concise
  # than the TensorBoard log files.
  os.makedirs(train_path, exist_ok=True)
  with open(os.path.join(train_path, "train.history.json"), "w") as fd:
    # Use LIT's custom JSON encoder to handle dicts containing NumPy data.
    fd.write(serialize.to_json(history, simple=True, indent=2))

  model.save(train_path)
  logging.info("Saved model files: \n  %s",
               "\n  ".join(os.listdir(train_path)))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  ##
  # Pick the model and datasets
  # TODO(lit-dev): add remaining GLUE tasks? These three cover all the major
  # features (single segment, two segment, classification, regression).
  if _TASK.value == "sst2":
    train_data = glue_data.SST2Data("train")
    val_data = glue_data.SST2Data("validation")
    model = glue_models.SST2Model(_ENCODER_NAME.value)
  elif _TASK.value == "mnli":
    train_data = glue_data.MNLIData("train")
    val_data = glue_data.MNLIData("validation_matched")
    model = glue_models.MNLIModel(_ENCODER_NAME.value)
  elif _TASK.value == "stsb":
    train_data = glue_data.STSBData("train")
    val_data = glue_data.STSBData("validation")
    model = glue_models.STSBModel(_ENCODER_NAME.value)
  else:
    raise ValueError(f"Unrecognized task name: '{_TASK.value:s}'")

  ##
  # Run training and save model.
  train_and_save(
      model,
      train_data,
      val_data,
      _TRAIN_PATH.value,
      save_intermediates=_SAVE_INTERMEDIATES.value,
      num_epochs=_NUM_EPOCHS.value)


if __name__ == "__main__":
  app.run(main)
