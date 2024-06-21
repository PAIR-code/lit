"""Methods for configuring models for prompt debugging."""

from collections.abc import Mapping, Sequence
import os
from typing import Optional

from absl import logging
from lit_nlp.api import model as lit_model
from lit_nlp.lib import file_cache


DEFAULT_BATCH_SIZE = 1
DEFAULT_DL_FRAMEWORK = "kerasnlp"
DEFAULT_DL_RUNTIME = "tensorflow"
DEFAULT_MODELS = ["gemma_1.1_instruct_2b_en:gemma_1.1_instruct_2b_en"]
DEFAULT_PRECISION = "bfloat16"


def _initialize_modeling_environment(
    dl_framework: str,
    dl_runtime: str,
    precision: str,
) -> None:
  """Configure the modeling environment."""
  if dl_framework == "kerasnlp":
    # NOTE: Keras requires that the KERAS_BACKEND variable is set before import.
    os.environ["KERAS_BACKEND"] = dl_runtime

    # NOTE: Imported here and not at the top of the file to avoid
    # initialization issues with the environment variables above.
    import keras  # pylint: disable=g-import-not-at-top

    keras.config.set_floatx(precision)
  elif dl_runtime == "torch":
    # NOTE: Keras sets precision for all backends with set_floatx(), but for
    # HuggingFace Transformers with PyTorch we need to set it explicitly.
    import torch  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    torch.set_default_dtype(
        torch.bfloat16 if precision == "bfloat16" else torch.float32
    )


def get_models(
    models_config: Optional[Sequence[str]] = None,
    dl_framework: str = DEFAULT_DL_FRAMEWORK,
    dl_runtime: str = DEFAULT_DL_RUNTIME,
    precision: str = DEFAULT_PRECISION,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Mapping[str, lit_model.Model]:
  """Loads models from the given configs.

  Args:
    models_config: A list of model names and paths to load from, as
      "model:path", where path can be a URL, a local file path, or the name of a
      preset for the configured deep learning framework.
    dl_framework: The deep learning framework that loads and runs the model on
      the runtime, `models_config.path` incompatibilities will result in errors.
    dl_runtime: The deep learning runtime that the model runs on, either
      "tensorflow" or "torch". All loaded models will use the same runtime,
      incompatibilities will result in errors.
    precision: Floating point precision for the models, either `bfloat16` or
      `float32`.
    batch_size: The number of examples to process per batch.

  Returns:
    A mapping from model name to initialized LIT model.
  """

  if not models_config:
    return {}

  # NOTE: Always call this function before initializing models to ensure the
  # environment is properly configured.
  _initialize_modeling_environment(dl_framework, dl_runtime, precision)

  models: dict[str, lit_model.Model] = {}
  for model_string in models_config:
    # Only split on the first ':' as path may be a URL containing 'https://'
    model_name, path = model_string.split(":", 1)
    logging.info("Loading model '%s' from '%s'", model_name, path)

    if path.endswith(".tar.gz") or file_cache.is_remote(path):
      path = file_cache.cached_path(
          path,
          extract_compressed_file=path.endswith(".tar.gz"),
      )

    if dl_framework == "kerasnlp":
      from lit_nlp.examples.prompt_debugging import keras_lms  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

      models |= keras_lms.initialize_model_group_for_salience(
          model_name,
          model_name_or_path=path,
          max_length=512,
          batch_size=batch_size,
      )
    else:
      from lit_nlp.examples.prompt_debugging import transformers_lms  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

      models |= transformers_lms.initialize_model_group_for_salience(
          model_name,
          path,
          batch_size=batch_size,
          framework=dl_runtime,
          max_new_tokens=512,
      )

  return models
