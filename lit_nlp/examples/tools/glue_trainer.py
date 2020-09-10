# Lint as: python3
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
AutoTokenizer, AutoConfig, and TFAutoModelForSequenceClassification, and can
load anything compatible with those classes.
"""
import os
from absl import app
from absl import flags
from absl import logging

from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models
from lit_nlp.lib import serialize
import tensorflow as tf

flags.DEFINE_string("encoder_name", "bert-base-uncased",
                    "Model name or path to pretrained (base) encoder.")
flags.DEFINE_string("task", "sst2", "Name of task to fine-tune on.")
flags.DEFINE_string("train_path", "/tmp/hf_demo",
                    "Path to save fine-tuned model.")

FLAGS = flags.FLAGS


def history_to_dict(keras_history):
  return {
      "epochs": keras_history.epoch,
      "history": keras_history.history,
      "params": keras_history.params,
      "optimizer_params": keras_history.model.optimizer.get_config(),
  }


def train_and_save(model, train_data, val_data, train_path):
  """Run training and save model."""
  # Set up logging for TensorBoard. To view, run:
  #   tensorboard --log_dir=<train_path>/tensorboard
  keras_callbacks = [
      tf.keras.callbacks.TensorBoard(
          log_dir=os.path.join(train_path, "tensorboard"))
  ]
  history = model.train(
      train_data.examples,
      validation_inputs=val_data.examples,
      keras_callbacks=keras_callbacks)

  # Save training history too, since this is human-readable and more concise
  # than the TensorBoard log files.
  with open(os.path.join(train_path, "train.history.json"), "w") as fd:
    # Use LIT's custom JSON encoder to handle dicts containing NumPy data.
    fd.write(serialize.to_json(history_to_dict(history), simple=True, indent=2))

  # Save model weights and config files.
  model.save(train_path)
  logging.info("Saved model files: \n  %s",
               "\n  ".join(os.listdir(train_path)))


def main(_):

  ##
  # Pick the model and datasets
  # TODO(lit-dev): add remaining GLUE tasks? These three cover all the major
  # features (single segment, two segment, classification, regression).
  if FLAGS.task == "sst2":
    train_data = glue.SST2Data("train")
    val_data = glue.SST2Data("validation")
    model = glue_models.SST2Model(FLAGS.encoder_name, for_training=True)
  elif FLAGS.task == "mnli":
    train_data = glue.MNLIData("train")
    val_data = glue.MNLIData("validation_matched")
    model = glue_models.MNLIModel(FLAGS.encoder_name, for_training=True)
  elif FLAGS.task == "stsb":
    train_data = glue.STSBData("train")
    val_data = glue.STSBData("validation")
    model = glue_models.STSBModel(FLAGS.encoder_name, for_training=True)
  else:
    raise ValueError(f"Unrecognized task name: '{FLAGS.task:s}'")

  ##
  # Run training and save model.
  train_and_save(model, train_data, val_data, FLAGS.train_path)


if __name__ == "__main__":
  app.run(main)
