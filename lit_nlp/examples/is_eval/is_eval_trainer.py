r"""Lightweight trainer script to fine-tune a model for IS eval.

Usage:
  python -m lit_nlp.examples.tools.is_eval_trainer \
    --encoder_name=bert-base-uncased \
    --train_path=/path/to/saved/model \
    --train_data_path=/path/to/train/data \
    --dev_data_path=/path/to/dev/data \

This will finetune a BERT model to reproduce findings of the paper ""Will You
Find These Shortcuts?" A Protocol for Evaluating the Faithfulness of Input
Salience Methods for Text Classification" [https://arxiv.org/abs/2111.07367].

Please ensure that the model's vocabulary file includes all special shortcut
tokens. When using the provided datasets of the LIT demo these are:
"ZEROA", "ZEROB", "ONEA", "ONEB", "onea", "oneb", "zeroa", "zerob", "synt".

This will train a BERT-base model [https://arxiv.org/abs/1810.04805]
which give validation accuracy in the low 90s on SST-2.

Note: you don't have to use this trainer to use LIT; the classifier
implementation is just a wrapper around HuggingFace Transformers, using
AutoTokenizer, AutoConfig, and TFAutoModelForSequenceClassification, and can
load anything compatible with those classes.
"""
import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from lit_nlp.examples.is_eval import datasets
from lit_nlp.examples.models import glue_models
from lit_nlp.lib import serialize
import tensorflow as tf

_ENCODER_NAME = flags.DEFINE_string(
    "encoder_name", "bert-base-uncased",
    "Model name or path to pretrained (base) encoder.")
_TRAIN_DATA_PATH = flags.DEFINE_string("train_data_path", None, "")
_DEV_DATA_PATH = flags.DEFINE_string("dev_data_path", None, "")
_TRAIN_PATH = flags.DEFINE_string("train_path", "/tmp/hf_demo",
                                  "Path to save fine-tuned model.")

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 3, "Number of epochs to train for.", lower_bound=1)
_SAVE_INTERMEDIATES = flags.DEFINE_bool(
    "save_intermediates", False,
    "If true, save intermediate weights after each epoch.")


def history_to_dict(keras_history):
  return {
      "epochs": keras_history.epoch,
      "history": keras_history.history,
      "params": keras_history.params,
      "optimizer_params": keras_history.model.optimizer.get_config(),
  }


class EpochSaverCallback(tf.keras.callbacks.Callback):
  """Save model at the beginning of training and after every epoch.

  Similar to tf.keras.callbacks.ModelCheckpoint, but this allows us to specify
  a custom save fn to call, such as the HuggingFace model.save() which writes
  .h5 files and config information.
  """

  def __init__(self, save_path_base: str, save_fn=None):
    super().__init__()
    self.save_path_base = save_path_base
    self.save_fn = save_fn or self.model.save

  def on_train_begin(self, logs=None):
    self.on_epoch_end(-1, logs=logs)  # write epoch-0

  def on_epoch_end(self, epoch, logs=None):
    # Save path 1-indexed = # of completed epochs.
    save_path = os.path.join(self.save_path_base, f"epoch-{epoch+1}")
    self.save_fn(save_path)


def train_and_save(model,
                   train_data,
                   val_data,
                   train_path,
                   save_intermediates=False,
                   **train_kw):
  """Run training and save model."""
  # Set up logging for TensorBoard. To view, run:
  #   tensorboard --log_dir=<train_path>/tensorboard
  keras_callbacks = [
      tf.keras.callbacks.TensorBoard(
          log_dir=os.path.join(train_path, "tensorboard"))
  ]
  if save_intermediates:
    keras_callbacks.append(EpochSaverCallback(train_path, save_fn=model.save))
  history = model.train(
      train_data.examples,
      validation_inputs=val_data.examples,
      keras_callbacks=keras_callbacks,
      **train_kw)

  # Save training history too, since this is human-readable and more concise
  # than the TensorBoard log files.
  with open(os.path.join(train_path, "train.history.json"), "w") as fd:
    # Use LIT's custom JSON encoder to handle dicts containing NumPy data.
    fd.write(serialize.to_json(history_to_dict(history), simple=True, indent=2))

  model.save(train_path)
  logging.info("Saved model files: \n  %s",
               "\n  ".join(os.listdir(train_path)))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  model = glue_models.SST2Model(_ENCODER_NAME.value)
  train_data = datasets.SingleInputClassificationFromTSV(_TRAIN_DATA_PATH.value)
  dev_data = datasets.SingleInputClassificationFromTSV(_DEV_DATA_PATH.value)

  train_and_save(
      model,
      train_data,
      dev_data,
      _TRAIN_PATH.value,
      save_intermediates=_SAVE_INTERMEDIATES.value,
      num_epochs=_NUM_EPOCHS.value,
      learning_rate=1e-5,
      batch_size=16,
  )


if __name__ == "__main__":
  app.run(main)
