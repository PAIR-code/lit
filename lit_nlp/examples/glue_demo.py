# Lint as: python3
r"""Example demo loading a handful of GLUE models.

To run locally:
  python -m lit_nlp.examples.glue_demo \
      --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""
import os

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "tasks", ["sst2", "stsb", "mnli"],
    "Tasks to include in this demo. See below for models and datasets this will load."
)

flags.DEFINE_string(
    "models_path", None,
    "Path to fine-tuned model files. Expects models to be in "
    "<models_path>/<task_name>, and in standard transformers format, e.g. as "
    "saved by model.save_pretrained() and tokenizer.save_pretrained().")

flags.DEFINE_integer(
    "max_examples", None, "Maximum number of examples to load into LIT. "
    "Note: MNLI eval set is 10k examples, so will take a while to run and may "
    "be slow on older machines. Set --max_examples=200 for a quick start.")


def main(_):

  models = {}
  datasets = {}

  if "sst2" in FLAGS.tasks:
    models["sst2"] = glue_models.SST2Model(
        os.path.join(FLAGS.models_path, "sst2"))
    datasets["sst_dev"] = glue.SST2Data("validation")
    logging.info("Loaded models and data for SST-2 task.")

  if "stsb" in FLAGS.tasks:
    models["stsb"] = glue_models.STSBModel(
        os.path.join(FLAGS.models_path, "stsb"))
    datasets["stsb_dev"] = glue.STSBData("validation")
    logging.info("Loaded models and data for STS-B task.")

  if "mnli" in FLAGS.tasks:
    models["mnli"] = glue_models.MNLIModel(
        os.path.join(FLAGS.models_path, "mnli"))
    datasets["mnli_dev"] = glue.MNLIData("validation_matched")
    datasets["mnli_dev_mm"] = glue.MNLIData("validation_mismatched")
    logging.info("Loaded models and data for MultiNLI task.")

  # Truncate datasets if --max_examples is set.
  for name in datasets:
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))
    datasets[name] = datasets[name].slice[:FLAGS.max_examples]
    logging.info("  truncated to %d examples", len(datasets[name]))

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
