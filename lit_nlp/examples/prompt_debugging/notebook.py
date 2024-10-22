"""Convenience functions for configuring LIT prompt debugging in a notebook."""

from collections.abc import Sequence

from lit_nlp import notebook as lit_notebook
from lit_nlp.examples.prompt_debugging import datasets
from lit_nlp.examples.prompt_debugging import layouts
from lit_nlp.examples.prompt_debugging import models


def make_notebook_widget(
    datasets_config: Sequence[str],
    models_config: Sequence[str],
    *,
    # keep-sorted start
    batch_size: int = models.DEFAULT_BATCH_SIZE,
    dl_framework: str = "kerasnlp",
    dl_runtime: str = "tensorflow",
    max_examples: int = datasets.DEFAULT_MAX_EXAMPLES,
    precision: str = "bfloat16",
    # keep-sorted end,
    **kwargs,
) -> lit_notebook.LitWidget:
  """Initializes a LIT widget for prompt debugging in a notebook.

  Args:
    datasets_config: A list of dataset names and paths to load from, as
      "dataset:path", where path can be a URL, a local file path, or the name of
      a preset for the configured deep learning framework.
    models_config: A list of model names and paths to load from, as
      "model:path", where path can be a URL, a local file path, or the name of a
      preset for the configured deep learning framework.
    batch_size: The number of examples the model will process per batch.
    dl_framework: The deep learning framework that loads and runs the model on
      the runtime, `models_config.path` incompatibilities will result in errors.
    dl_runtime: The deep learning runtime that the model runs on, either
      "tensorflow" or "torch". All loaded models will use the same runtime,
      incompatibilities will result in errors.
    max_examples: Maximum number of examples in each loaded dataset.
    precision: Floating point precision for the models, either `bfloat16` or
      `float32`.
    **kwargs: Additional keyword arguments passed to the LitWidget. See also
      LitApp for additinoal keyword arguments accepted by the LitWidget.

  Returns:
    A LitWidget with the configured models and datasets. Call `widget.render()`
    to load the data and render the UI.
  """
  return lit_notebook.LitWidget(
      models=models.get_models(
          models_config=models_config,
          dl_framework=dl_framework,
          dl_runtime=dl_runtime,
          precision=precision,
          batch_size=batch_size,
      ),
      datasets=datasets.get_datasets(
          datasets_config=datasets_config, max_examples=max_examples
      ),
      layouts=layouts.PROMPT_DEBUGGING_LAYOUTS,
      default_layout=layouts.LEFT_RIGHT,
      model_loaders=models.get_model_loaders(
          dl_framework=dl_framework,
          dl_runtime=dl_runtime,
          batch_size=batch_size,
          max_length=models.DEFAULT_SEQUENCE_LENGTH,
      ),
      dataset_loaders=datasets.get_dataset_loaders(),
      **kwargs,
  )
