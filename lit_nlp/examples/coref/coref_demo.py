r"""Coreference demo, trainer and LIT server.

To run LIT locally with a pre-trained model:
  blaze run -c opt --config=cuda examples/coref:coref_demo -- --port=5432

This demo shows a simple gold-mention coreference model, highlighting how LIT
can support intersectional analysis for fairness evaluation. It also
demonstrates a multi-headed model, with both a structured prediction head
and a two-class classifier to predict the binary answer on Winogender examples.

Our model is a probing-style classifier over a frozen BERT encoder, similar to
Tenney et al. 2019 (https://arxiv.org/abs/1905.06316). This model takes gold
mention pairs as input and predicts binary labels - 1 if coreferent, else 0.
This is not the best coference model one could build, but it's a good one if
we're interested in probing the kinds of biases encoded in a language model
such as BERT. Plus, because we can pre-compute the BERT activations, we can
train our model very quickly.

For evaluation, we use the Winogender dataset of Rudinger et al. 2018
(https://arxiv.org/abs/1804.09301), which consists of 720 template-generated
sentences, each with a pronoun and two candidate mentions: an occupation term,
and an neutral participant such as 'customer'. Each instance is annotated with
pf_bls, the fraction of that occupation identifying as female, per the U.S.
Bureau of Labor Statistics.

For more details on the analysis, see the case study in Section 3 of
the LIT paper (https://arxiv.org/abs/2008.05122).

To train the model, you'll need the OntoNotes 5.0 dataset in the edge probing
JSON format. See
https://github.com/nyu-mll/jiant-v1-legacy/tree/master/probing/data#ontonotes
for instructions. Then run:
  blaze run -c opt --config=cuda examples/coref:coref_demo -- \
    --encoder_name=bert-base-uncased --do_train \
    --ontonotes_edgeprobe_path=/path/to/ontonotes/coref/ \
    --model_path=/path/to/save/model \
    --do_serve --port=5432

With bert-base-uncased on a single Titan Xp GPU, it takes about 10-12 minutes
to train this model, including the time to extract representations, and should
get around 85% F1 on the OntoNotes development set. Exact numbers on Winogender
will vary, but the qualitative behavior on slices by gender and answer should
match Figure 3 of the paper.
"""
import copy
import os
import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes as lit_dtypes
from lit_nlp.api import layout
from lit_nlp.api import types as lit_types
from lit_nlp.examples.coref import edge_predictor
from lit_nlp.examples.coref import encoders
from lit_nlp.examples.coref import model
from lit_nlp.examples.coref.datasets import ontonotes
from lit_nlp.examples.coref.datasets import winogender
from lit_nlp.lib import utils

import transformers  # for path caching

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_DO_TRAIN = flags.DEFINE_bool(
    "do_train", False,
    "If true, train a new model and save to FLAGS.model_path.")
_DO_SERVE = flags.DEFINE_bool(
    "do_serve", True,
    "If true, start a LIT server with the model at FLAGS.model_path.")

_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    "https://storage.googleapis.com/what-if-tool-resources/lit-models/coref_base.tar.gz",
    "Path to save or load trained model.")

##
# Training-only flags; these are ignored if only serving a pre-trained model.
_ENCODER_NAME = flags.DEFINE_string(
    "encoder_name", "bert-base-uncased",
    "Name of BERT variant to use for fine-tuning. See https://huggingface.co/models."
)

_ONTONOTES_EDGEPROBE_PATH = flags.DEFINE_string(
    "ontonotes_edgeprobe_path", None,
    "Path to OntoNotes coreference data in edge probing JSON format. "
    "This is needed for training, and optional for running LIT.")

# Custom frontend layout; see client/lib/types.ts
modules = layout.LitModuleName
WINOGENDER_LAYOUT = layout.LitComponentLayout(
    components={
        "Main": [
            modules.DataTableModule,
            modules.DatapointEditorModule,
        ],
        "Predictions": [
            modules.SpanGraphGoldModule,
            modules.SpanGraphModule,
            modules.ClassificationModule,
        ],
        "Performance": [
            modules.MetricsModule,
            modules.ScalarModule,
            modules.ConfusionMatrixModule,
        ],
    },
    description="Custom layout for the Winogender coreference demo.",
)
CUSTOM_LAYOUTS = {"winogender": WINOGENDER_LAYOUT}

FLAGS.set_default("default_layout", "winogender")


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Return WSGI app for container-hosted demos."""
  # Set defaults for container-hosted demo.
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("do_train", False)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("coref_demo:get_wsgi_app() called with unused args: %s",
                 unused)
  return main([])


def symmetrize_edges(dataset: lit_dataset.Dataset) -> lit_dataset.Dataset:
  """Symmetrize edges by adding copies with span1 and span2 interchanged."""

  def _swap(edge):
    return lit_dtypes.EdgeLabel(edge.span2, edge.span1, edge.label)

  edge_fields = utils.find_spec_keys(dataset.spec(), lit_types.EdgeLabels)
  examples = []
  for ex in dataset.examples:
    new_ex = copy.copy(ex)
    for field in edge_fields:
      new_ex[field] += [_swap(edge) for edge in ex[field]]
    examples.append(new_ex)
  return lit_dataset.Dataset(dataset.spec(), examples)


def train(save_path: str):
  """Train a coreference model using encoder features over OntoNotes."""
  # Load OntoNotes data for training.
  ontonotes_train = ontonotes.OntonotesCorefDataset(
      os.path.join(FLAGS.ontonotes_edgeprobe_path, "train.json"))
  ontonotes_dev = ontonotes.OntonotesCorefDataset(
      os.path.join(FLAGS.ontonotes_edgeprobe_path, "development.json"))

  # Assemble our model.
  encoder = encoders.BertEncoderWithOffsets(FLAGS.encoder_name)
  input_dim = encoder.model.config.hidden_size
  classifier = edge_predictor.SingleEdgePredictor(
      input_dim=input_dim, hidden_dim=min(input_dim, 256))
  full_model = model.FrozenEncoderCoref(encoder, classifier)

  # Train our model.
  train_dataset = symmetrize_edges(ontonotes_train)
  full_model.train(
      train_dataset.examples,
      ontonotes_dev.examples,
      batch_size=128,
      num_epochs=15)
  # Save classifier and encoder
  full_model.save(save_path)


def run_server(load_path: str):
  """Run a LIT server with the trained coreference model."""
  # Normally path is a directory; if it's an archive file, download and
  # extract to the transformers cache.
  if load_path.endswith(".tar.gz"):
    load_path = transformers.file_utils.cached_path(
        load_path, extract_compressed_file=True)
  # Load model from disk.
  full_model = model.FrozenEncoderCoref.from_saved(
      load_path,
      encoder_cls=encoders.BertEncoderWithOffsets,
      classifier_cls=edge_predictor.SingleEdgePredictor)

  # Set up the LIT server.
  models = {"model": full_model}
  datasets = {"winogender": winogender.WinogenderDataset()}
  if _ONTONOTES_EDGEPROBE_PATH.value:
    datasets["ontonotes_dev"] = ontonotes.OntonotesCorefDataset(
        os.path.join(FLAGS.ontonotes_edgeprobe_path, "development.json"))
  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(
      models, datasets, layouts=CUSTOM_LAYOUTS, **server_flags.get_flags())
  return lit_demo.serve()


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  assert _MODEL_PATH.value, "Must specify --model_path to run."

  if _DO_TRAIN.value:
    train(_MODEL_PATH.value)

  if _DO_SERVE.value:
    return run_server(_MODEL_PATH.value)


if __name__ == "__main__":
  app.run(main)
