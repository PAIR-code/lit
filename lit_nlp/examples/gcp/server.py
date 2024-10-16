# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Server for sequence salience with a left-to-right language model."""

from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.components import scrambler
from lit_nlp.components import word_replacer
from lit_nlp.examples.gcp import model as lit_gcp_model
from lit_nlp.examples.gcp import vertexai_models
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

  datasets = pd_datasets.get_datasets(
      datasets_config=pd_datasets.DEFAULT_DATASETS,
      max_examples=pd_datasets.DEFAULT_MAX_EXAMPLES,
  )

  model_loaders = lit_gcp_model.get_model_loaders()
  model_loaders["gemini"] = (
      vertexai_models.GeminiFoundationalModel,
      vertexai_models.GeminiFoundationalModel.init_spec(),
  )

  generators = {
      "word_replacer": word_replacer.WordReplacer(),
      "scrambler": scrambler.Scrambler(),
  }

  lit_demo = dev_server.Server(
      models={},
      datasets=datasets,
      layouts=pd_layouts.PROMPT_DEBUGGING_LAYOUTS,
      model_loaders=model_loaders,
      generators=generators,
      dataset_loaders=pd_datasets.get_dataset_loaders(),
      onboard_start_doc=_SPLASH_SCREEN_DOC,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
