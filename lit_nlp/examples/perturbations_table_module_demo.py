# Lint as: python3
r"""Testing server for the PerturbationsTable UI module.

To run locally:
  python -m lit_nlp.examples.perturbations_table_module_demo \
      --port=5432

Once you see the ASCII-art LIT logo, navigate to localhost:5432 to access
the demo UI.  No training required.
"""
import tempfile

from absl import app

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models

def main(_):
  max_examples = 222
  models = {
    "sst_distilbert": glue_models.SST2Model("distilbert-base-uncased-finetuned-sst-2-english"),
    "sst_roberta": glue_models.SST2Model("textattack/roberta-base-SST-2"),
    "mnli_distilbert": glue_models.MNLIModel("huggingface/distilbert-base-uncased-finetuned-mnli"),
    "stsb_distilbert": glue_models.STSBModel("sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking")
  }
  datasets = {
    "sst_dev": glue.SST2Data("validation").sample(max_examples),
    "mnli_validation_matched": glue.MNLIData("validation_matched").sample(max_examples),
    "stsb_dev": glue.STSBData("validation").sample(max_examples)
  }

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
