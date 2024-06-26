r"""A blank demo ready to load models and datasets.

The currently supported models and datasets are:
- classification model on SST-2, with the Stanford Sentiment Treebank dataset.
- regression model on STS-B, with Semantic Textual Similarit Benchmark dataset.
- classification model on MultiNLI, with the MultiNLI dataset.
- TensorFlow Keras model for penguin classification, with the Penguin tabular
  dataset from TFDS.

To run:
  python -m lit_nlp.examples.blank_slate_demo --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""

from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from lit_nlp import app as lit_app
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.glue import data as glue_data
from lit_nlp.examples.glue import models as glue_models
from lit_nlp.examples.penguin import data as penguin_data
from lit_nlp.examples.penguin import model as penguin_model

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Return WSGI app for container-hosted demos."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info(
        "blank_slate_demo:get_wsgi_app() called with unused args: %s", unused
    )
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  models = {}
  model_loaders: lit_app.ModelLoadersMap = {}

  # glue demo model loaders.
  model_loaders["sst2"] = (
      glue_models.SST2Model,
      glue_models.GlueModelConfig.init_spec(),
  )
  model_loaders["stsb"] = (
      glue_models.STSBModel,
      glue_models.GlueModelConfig.init_spec(),
  )
  model_loaders["mnli"] = (
      glue_models.MNLIModel,
      glue_models.GlueModelConfig.init_spec(),
  )

  # penguin demo model loaders.
  model_loaders["penguin"] = (
      penguin_model.PenguinModel,
      penguin_model.PenguinModel.init_spec(),
  )

  datasets = {}
  dataset_loaders: lit_app.DatasetLoadersMap = {}

  # glue demo dataset loaders.
  dataset_loaders["sst2"] = (glue_data.SST2Data, glue_data.SST2Data.init_spec())
  dataset_loaders["stsb"] = (glue_data.STSBData, glue_data.STSBData.init_spec())
  dataset_loaders["mnli"] = (glue_data.MNLIData, glue_data.MNLIData.init_spec())

  # penguin demo dataset loaders.
  dataset_loaders["penguin"] = (
      penguin_data.PenguinDataset,
      penguin_data.PenguinDataset.init_spec(),
  )

  # lm demo dataset loaders.
  dataset_loaders["sst (lm)"] = (
      glue_data.SST2DataForLM,
      glue_data.SST2DataForLM.init_spec(),
  )

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(
      models,
      datasets,
      model_loaders=model_loaders,
      dataset_loaders=dataset_loaders,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
