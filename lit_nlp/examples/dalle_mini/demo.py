r"""Example for dalle-mini demo model.

First run following command to install required packages:
  pip install -r ./lit_nlp/examples/dalle_mini/requirements.txt

To run locally with a small number of examples:
  python -m lit_nlp.examples.dalle_mini.demo

By default, this module uses the "cuda" device for image generation.
The `requirements.txt` file installs a CUDA-enabled version of PyTorch for GPU
acceleration.

If you are running on a machine without a compatible GPU or CUDA drivers,
you must switch the device to "cpu" and reinstall the CPU-only version of
PyTorch.

Usage:
    - Default: device="cuda"
    - On CPU-only machines:
        1. Set device="cpu" during model initialization
        2. Uninstall the CUDA version of PyTorch:
               pip uninstall torch
        3. Install the CPU-only version:
               pip install torch==2.1.2+cpu --extra-index-url
               https://download.pytorch.org/whl/cpu

Example:
    >>> model = MinDalle(..., device="cpu")

Check CUDA availability:
    >>> import torch
    >>> torch.cuda.is_available()
    False  # if no GPU support is present

Error Handling:
    - If CUDA is selected but unsupported, you will see:
          AssertionError: Torch not compiled with CUDA enabled
    - To fix this, either install the correct CUDA-enabled PyTorch or switch to
    CPU mode.

Then navigate to localhost:5432 to access the demo UI.
"""

from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from lit_nlp import app as lit_app
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.dalle_mini import data as dalle_data
from lit_nlp.examples.dalle_mini import model as dalle_model


# NOTE: additional flags defined in server_flags.py
_FLAGS = flags.FLAGS
_FLAGS.set_default("development_demo", True)
_FLAGS.set_default("default_layout", "DALLE_LAYOUT")

_MODELS = (["dalle-mini"],)

_CANNED_PROMPTS = ["I have a dream", "I have a shiba dog named cola"]

# Custom frontend layout; see api/layout.py
_modules = layout.LitModuleName
_DALLE_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [
            _modules.DataTableModule,
            _modules.DatapointEditorModule,
        ]
    },
    lower={
        "Predictions": [
            _modules.GeneratedImageModule,
            _modules.GeneratedTextModule,
        ],
    },
    description="Custom layout for Text to Image models.",
)


CUSTOM_LAYOUTS = layout.DEFAULT_LAYOUTS | {"DALLE_LAYOUT": _DALLE_LAYOUT}


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  _FLAGS.set_default("server_type", "external")
  _FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = _FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load models, according to the --models flag.
  models = {}

  model_loaders: lit_app.ModelLoadersMap = {}
  model_loaders["dalle-mini"] = (
      dalle_model.DalleMiniModel,
      dalle_model.DalleMiniModel.init_spec(),
  )

  datasets = {"examples": dalle_data.DallePrompts(_CANNED_PROMPTS)}
  dataset_loaders: lit_app.DatasetLoadersMap = {}
  dataset_loaders["text_to_image"] = (
      dalle_data.DallePrompts,
      dalle_data.DallePrompts.init_spec(),
  )

  lit_demo = dev_server.Server(
      models=models,
      model_loaders=model_loaders,
      datasets=datasets,
      dataset_loaders=dataset_loaders,
      layouts=CUSTOM_LAYOUTS,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
