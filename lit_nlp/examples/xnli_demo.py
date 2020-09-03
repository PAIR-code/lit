# Lint as: python3
r"""Example demo for multilingual NLI on the XNLI eval set.

To train a model for this task, use tools/glue_trainer.py or your favorite
trainer script to fine-tune a multilingual encoder, such as
bert-base-multilingual-cased, on the mnli task.

To run locally:
  python -m lit_nlp.examples.xnli_demo \
      --model_path=/path/to/multilingual/mnli/model \
      --port=5432

Then navigate to localhost:5432 to access the demo UI.

Note: the LIT UI can handle around 10k examples comfortably, depending on your
hardware. The monolingual (english) eval sets for MNLI are about 9.8k each,
while each language for XNLI is about 2.5k examples, so we recommend using the
--languages flag to load only the languages you're interested in.
"""
from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import classification
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "languages", ["en", "es", "hi", "zh"],
    "Languages to load from XNLI. Available languages: "
    "ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,zh,vi")

flags.DEFINE_string(
    "model_path", None,
    "Path to fine-tuned model files. Expects model to be in standard "
    "transformers format, e.g. as saved by model.save_pretrained() and "
    "tokenizer.save_pretrained().")

flags.DEFINE_integer(
    "max_examples", None, "Maximum number of examples to load into LIT. "
    "Note: MNLI eval set is 10k examples, so will take a while to run and may "
    "be slow on older machines. Set --max_examples=200 for a quick start.")


def main(_):

  models = {
      "nli": glue_models.MNLIModel(FLAGS.model_path, inference_batch_size=16)
  }
  datasets = {
      "xnli": classification.XNLIData("validation", FLAGS.languages),
      "mnli_dev": glue.MNLIData("validation_matched"),
      "mnli_dev_mm": glue.MNLIData("validation_mismatched"),
  }

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
