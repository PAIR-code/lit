r"""An blank demo ready to load models and datasets.

The currently supported models and datasets are:
- classification model on SST-2, with the Stanford Sentiment Treebank dataset.
- regression model on STS-B, with Semantic Textual Similarit Benchmark dataset.
- classification model on MultiNLI, with the MultiNLI dataset.
- TensorFlow Keras model for penguin classification, with the Penguin tabular
  dataset from TFDS.
- T5 models using HuggingFace Transformers and Keras, with the English CNNDM
  summarization dataset and the WMT '14 machine-translation dataset.
- BERT (bert-*) as a masked language model and GPT-2 (gpt2* or distilgpt2) as a
  left-to-right language model, with the Stanford Sentiment Treebank dataset,
  the IMDB reviews dataset, Billion Word Benchmark (lm1b) dataset and the option
  to load sentences from a flat text file.
- MobileNet model, with the Imagenette TFDS dataset.

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
from lit_nlp.examples.datasets import classification
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.datasets import imagenette
from lit_nlp.examples.datasets import lm
from lit_nlp.examples.datasets import mt
from lit_nlp.examples.datasets import penguin_data
from lit_nlp.examples.datasets import summarization
from lit_nlp.examples.models import glue_models
from lit_nlp.examples.models import mobilenet
from lit_nlp.examples.models import penguin_model
from lit_nlp.examples.models import pretrained_lms
from lit_nlp.examples.models import t5

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

  # t5 demo model loaders.
  model_loaders["T5 summarization"] = (
      t5.T5Summarization,
      t5.T5Summarization.init_spec(),
  )
  model_loaders["T5 translation"] = (
      t5.T5Translation,
      t5.T5Translation.init_spec(),
  )

  # lm demo model loaders.
  model_loaders["bert"] = (
      pretrained_lms.BertMLM,
      pretrained_lms.BertMLM.init_spec(),
  )
  model_loaders["gpt2"] = (
      pretrained_lms.GPT2LanguageModel,
      pretrained_lms.GPT2LanguageModel.init_spec(),
  )

  # image model loaders.
  model_loaders["image"] = (
      mobilenet.MobileNet,
      mobilenet.MobileNet.init_spec(),
  )

  datasets = {}
  dataset_loaders: lit_app.DatasetLoadersMap = {}

  # glue demo dataset loaders.
  dataset_loaders["sst2"] = (glue.SST2Data, glue.SST2Data.init_spec())
  dataset_loaders["stsb"] = (glue.STSBData, glue.STSBData.init_spec())
  dataset_loaders["mnli"] = (glue.MNLIData, glue.MNLIData.init_spec())

  # penguin demo dataset loaders.
  dataset_loaders["penguin"] = (
      penguin_data.PenguinDataset,
      penguin_data.PenguinDataset.init_spec(),
  )

  # t5 demo dataset loaders.
  dataset_loaders["CNN DailyMail (t5)"] = (
      summarization.CNNDMData,
      summarization.CNNDMData.init_spec(),
  )
  dataset_loaders["WMT 14 (t5)"] = (mt.WMT14Data, mt.WMT14Data.init_spec())

  # lm demo dataset loaders.
  dataset_loaders["sst (lm)"] = (
      glue.SST2DataForLM,
      glue.SST2DataForLM.init_spec(),
  )
  dataset_loaders["imdb (lm)"] = (
      classification.IMDBData,
      classification.IMDBData.init_spec(),
  )
  dataset_loaders["plain text sentences (lm)"] = (
      lm.PlaintextSents,
      lm.PlaintextSents.init_spec(),
  )
  dataset_loaders["bwb (lm)"] = (
      lm.BillionWordBenchmark,
      lm.BillionWordBenchmark.init_spec(),
  )

  # image demo dataset loaders.
  dataset_loaders["image"] = (
      imagenette.ImagenetteDataset,
      imagenette.ImagenetteDataset.init_spec(),
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
