# Lint at: python3
"""Example demo loading Stanza models.

To run with the demo:
    python -m lit_nlp.examples.stanza_demo --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""
from absl import app
from absl import flags

import lit_nlp.api.dataset as lit_dataset
import lit_nlp.api.types as lit_types
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import stanza_models
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.components import scrambler
from lit_nlp.components import word_replacer

FLAGS = flags.FLAGS

flags.DEFINE_list(
  "sequence_tasks",
  ["upos", "xpos", "lemma"],
  "Sequence tasks to load and use for prediction. Defaults to all sequence tasks",
)

flags.DEFINE_list(
  "span_tasks",
  ["mention"],
  "Span tasks to load and use for prediction. Only mentions are included in this demo",
)

flags.DEFINE_list(
  "edge_tasks",
  ["deps"],
  "Span tasks to load and use for prediction. Only deps are included in this demo",
)

flags.DEFINE_string("language", "en", "Language to load for Stanza model.")

flags.DEFINE_integer(
  "max_examples", None, "Maximum number of examples to load into LIT."
)


def main(_):
  # Set Tasks
  tasks = {
    "sequence": FLAGS.sequence_tasks,
    "span": FLAGS.span_tasks,
    "edge": FLAGS.edge_tasks,
  }

  # Get the correct model for the language
  stanza.download(FLAGS.language)
  pretrained_model = stanza.Pipeline(FLAGS.language)
  models = {
    "stanza": stanza_models.StanzaTagger(pretrained_model, tasks),
  }

  # Datasets for LIT demo
  datasets = {
    "SST2": glue.SST2Data(split="validation").slice[: FLAGS.max_examples],
    "blank": lit_dataset.Dataset({"text": lit_types.TextSegment()}, []),
  }

  # Add generators
  generators = {
    "scrambler": scrambler.Scrambler(),
    "word_replacer": word_replacer.WordReplacer(),
  }

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(
    models, datasets, generators, **server_flags.get_flags()
  )
  lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
