# Lint as: python3
r"""Example demo loading a T5 model for a summarization task.

To run locally:
  python -m lit_nlp.examples.t5_summarization_demo \
      --port=5432 --warm_start 1.0 --top_k 10 --use_indexer --initialize_index \
      --data_dir=/tmp/t5_index

Then navigate to localhost:5432 to access the demo UI.
"""
import os
import sys

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.components import index
from lit_nlp.components import similarity_searcher
from lit_nlp.components import word_replacer
from lit_nlp.examples.datasets import summarization
from lit_nlp.examples.models import t5

# NOTE: additional flags defined in server_flags.py

flags.DEFINE_integer(
    "max_examples", 200,
    "Maximum number of examples to load from the development set.")

flags.DEFINE_integer("max_index_examples", 2000,
                     "Maximum number of examples to index from the train set.")

flags.DEFINE_list("models", ["t5-small"], "Which model(s) to load.")

flags.DEFINE_integer("top_k", 10,
                     "Rank to which the output distribution is pruned.")

flags.DEFINE_boolean("use_indexer", True,
                     "If true, will use the nearest neighbor index.")
flags.DEFINE_boolean(
    "initialize_index", True,
    "If the flag is set, it builds the nearest neighbor index before starting "
    "the server. If false, will look for one in --data_dir. No effect if "
    "--use_indexer is False.")

FLAGS = flags.FLAGS


def get_wsgi_app():
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("data_dir", "./t5_data/")
  FLAGS.set_default("initialize_index", False)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(_):
  ##
  # Load models. You can specify several here, if you want to compare different
  # models side-by-side, and can also include models of different types that use
  # different datasets.
  models = {}
  for model_name_or_path in FLAGS.models:
    # Ignore path prefix, if using /path/to/<model_name> to load from a
    # specific directory rather than the default shortcut.
    model_name = os.path.basename(model_name_or_path)
    # TODO(lit-dev): attention is temporarily disabled, because O(n^2) between
    # tokens in a long document can get very, very large. Re-enable once we can
    # send this to the frontend more efficiently.
    models[model_name] = t5.T5GenerationModel(
        model_name=model_name_or_path,
        input_prefix="summarize: ",
        top_k=FLAGS.top_k,
        output_attention=False)

  ##
  # Load datasets. Typically you"ll have the constructor actually read the
  # examples and do any pre-processing, so that they"re in memory and ready to
  # send to the frontend when you open the web UI.
  datasets = {
      "CNNDM":
          summarization.CNNDMData(
              split="validation", max_examples=FLAGS.max_examples),
  }
  for name, ds in datasets.items():
    logging.info("Dataset: '%s' with %d examples", name, len(ds))

  ##
  # We can also add custom components. Generators are used to create new
  # examples by perturbing or modifying existing ones.
  generators = {
      # Word-substitution, like "great" -> "terrible"
      "word_replacer": word_replacer.WordReplacer(),
  }

  if FLAGS.use_indexer:
    assert FLAGS.data_dir, "--data_dir must be set to use the indexer."
    # Datasets for indexer - this one loads the training corpus instead of val.
    index_datasets = {
        "CNNDM":
            summarization.CNNDMData(
                split="train", max_examples=FLAGS.max_index_examples),
    }
    # Set up the Indexer, building index if necessary (this may be slow).
    indexer = index.Indexer(
        datasets=index_datasets,
        models=models,
        data_dir=FLAGS.data_dir,
        initialize_new_indices=FLAGS.initialize_index)

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
