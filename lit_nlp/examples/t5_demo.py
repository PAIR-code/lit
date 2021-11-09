# Lint as: python3
r"""Example demo loading a T5 model.

To run locally with a small number of examples:
  python -m lit_nlp.examples.t5_demo \
      --alsologtostderr --port=5432 --max_examples=10 \
      --nouse_indexer

To run using the nearest-neighbor lookup index (warning, this will take a while
to load):
  python -m lit_nlp.examples.t5_demo \
      --alsologtostderr --port=5432 --warm_start 1.0 \
      --use_indexer --initialize_index --data_dir=/tmp/t5_index

Then navigate to localhost:5432 to access the demo UI.
"""
import os
import sys

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.components import index
from lit_nlp.components import similarity_searcher
from lit_nlp.components import word_replacer
from lit_nlp.examples.datasets import mt
from lit_nlp.examples.datasets import summarization
from lit_nlp.examples.models import t5
from lit_nlp.lib import caching  # for hash id fn

# NOTE: additional flags defined in server_flags.py

flags.DEFINE_integer(
    "max_examples", 200,
    "Maximum number of examples to load from the development set.")

flags.DEFINE_integer("max_index_examples", 2000,
                     "Maximum number of examples to index from the train set.")

flags.DEFINE_list("models", ["t5-small"], "Which model(s) to load.")
flags.DEFINE_list("tasks", ["summarization", "mt"], "Which task(s) to load.")

flags.DEFINE_integer("token_top_k", 10,
                     "Rank to which the output distribution is pruned.")
flags.DEFINE_integer("num_to_generate", 4,
                     "Number of generations to produce for each input.")

##
# Options for nearest-neighbor indexer.
flags.DEFINE_boolean("use_indexer", True,
                     "If true, will use the nearest neighbor index.")
flags.DEFINE_boolean(
    "initialize_index", True,
    "If the flag is set, it builds the nearest neighbor index before starting "
    "the server. If false, will look for one in --data_dir. No effect if "
    "--use_indexer is False.")

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)


def get_wsgi_app():
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  FLAGS.set_default("data_dir", "./t5_data/")
  FLAGS.set_default("initialize_index", False)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)


def build_indexer(models):
  """Build nearest-neighbor indices."""
  assert FLAGS.data_dir, "--data_dir must be set to use the indexer."
  # Datasets for indexer - this one loads the training corpus instead of val.
  index_datasets = {
      "CNNDM":
          summarization.CNNDMData(
              split="train", max_examples=FLAGS.max_index_examples),
  }
  index_datasets = lit_dataset.IndexedDataset.index_all(index_datasets,
                                                        caching.input_hash)
  # TODO(lit-dev): add training data and indexing for MT task. This will be
  # easier after we remap the model specs, so it doesn't try to cross-index
  # between the summarization model and the MT data.
  index_models = {
      k: m for k, m in models.items() if isinstance(m, t5.SummarizationWrapper)
  }
  # Set up the Indexer, building index if necessary (this may be slow).
  return index.Indexer(
      datasets=index_datasets,
      models=index_models,
      data_dir=FLAGS.data_dir,
      initialize_new_indices=FLAGS.initialize_index)


def main(_):
  ##
  # Load models. You can specify several here, if you want to compare different
  # models side-by-side, and can also include models of different types that use
  # different datasets.
  base_models = {}
  for model_name_or_path in FLAGS.models:
    # Ignore path prefix, if using /path/to/<model_name> to load from a
    # specific directory rather than the default shortcut.
    model_name = os.path.basename(model_name_or_path)
    if model_name_or_path.startswith("SavedModel"):
      saved_model_path = model_name_or_path.split(":", 1)[1]
      base_models[model_name] = t5.T5SavedModel(saved_model_path)
    else:
      # TODO(lit-dev): attention is temporarily disabled, because O(n^2) between
      # tokens in a long document can get very, very large. Re-enable once we
      # can send this to the frontend more efficiently.
      base_models[model_name] = t5.T5HFModel(
          model_name=model_name_or_path,
          num_to_generate=FLAGS.num_to_generate,
          token_top_k=FLAGS.token_top_k,
          output_attention=False)

  ##
  # Load eval sets and model wrappers for each task.
  # Model wrappers share the same in-memory T5 model, but add task-specific pre-
  # and post-processing code.
  models = {}
  datasets = {}

  if "summarization" in FLAGS.tasks:
    for k, m in base_models.items():
      models[k + "_summarization"] = t5.SummarizationWrapper(m)
    datasets["CNNDM"] = summarization.CNNDMData(
        split="validation", max_examples=FLAGS.max_examples)

  if "mt" in FLAGS.tasks:
    for k, m in base_models.items():
      models[k + "_translation"] = t5.TranslationWrapper(m)
    datasets["wmt14_enfr"] = mt.WMT14Data(version="fr-en", reverse=True)
    datasets["wmt14_ende"] = mt.WMT14Data(version="de-en", reverse=True)

  # Truncate datasets if --max_examples is set.
  for name in datasets:
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))
    datasets[name] = datasets[name].slice[:FLAGS.max_examples]
    logging.info("  truncated to %d examples", len(datasets[name]))

  ##
  # We can also add custom components. Generators are used to create new
  # examples by perturbing or modifying existing ones.
  generators = {
      # Word-substitution, like "great" -> "terrible"
      "word_replacer": word_replacer.WordReplacer(),
  }

  if FLAGS.use_indexer:
    indexer = build_indexer(models)
    # Wrap the indexer into a Generator component that we can query.
    generators["similarity_searcher"] = similarity_searcher.SimilaritySearcher(
        indexer=indexer)

  ##
  # Actually start the LIT server, using the models, datasets, and other
  # components constructed above.
  lit_demo = dev_server.Server(
      models, datasets, generators=generators, **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
