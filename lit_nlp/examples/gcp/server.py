"""Server for sequence salience with a left-to-right language model."""

from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.gcp import model as lit_gcp_model
from lit_nlp.examples.prompt_debugging import datasets as pd_datasets
from lit_nlp.examples.prompt_debugging import layouts as pd_layouts


_FLAGS = flags.FLAGS

_SPLASH_SCREEN_DOC = """
# Language Model Salience

To begin, select an example, then click the segment(s) (tokens, words, etc.)
of the output that you would like to explain. Preceding segments(s) will be
highlighted according to their importance to the selected target segment(s),
with darker colors indicating a greater influence (salience) of that segment on
the model's likelihood of the target segment.
"""


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Return WSGI app for container-hosted demos."""
  _FLAGS.set_default("server_type", "external")
  _FLAGS.set_default("demo_mode", False)
  _FLAGS.set_default("page_title", "LM Prompt Debugging")
  _FLAGS.set_default("default_layout", pd_layouts.THREE_PANEL)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("lm_demo:get_wsgi_app() called with unused args: %s", unused)
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  lit_demo = dev_server.Server(
      models={},
      datasets={},
      layouts=pd_layouts.PROMPT_DEBUGGING_LAYOUTS,
      model_loaders=lit_gcp_model.get_model_loaders(),
      dataset_loaders=pd_datasets.get_dataset_loaders(),
      onboard_start_doc=_SPLASH_SCREEN_DOC,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)