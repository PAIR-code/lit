"""Server for sequence salience with a left-to-right language model."""

from collections.abc import Mapping, Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
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


def init_llm_on_gcp(
    name: str, base_url: str, *args, **kw
) -> Mapping[str, lit_model.Model]:
  return lit_gcp_model.initialize_model_group_for_salience(
      name=name, base_url=base_url, *args, **kw
  )


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Return WSGI app for container-hosted demos."""
  _FLAGS.set_default("server_type", "external")
  _FLAGS.set_default("demo_mode", True)
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
      model_loaders={
        'LLM on GCP': (init_llm_on_gcp, {
            'name': lit_types.String(),
            'base_url': lit_types.String(),
            'max_concurrent_requests': lit_types.Integer(default=1),
            'max_qps': lit_types.Scalar(default=25),
        })
      },
      dataset_loaders=pd_datasets.get_dataset_loaders(),
      onboard_start_doc=_SPLASH_SCREEN_DOC,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
