"""Methods for configuring models for prompt debugging."""

from collections.abc import Sequence
from typing import Optional

from absl import logging
from lit_nlp import app as lit_app
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types


DEFAULT_BATCH_SIZE = 1
DEFAULT_DL_RUNTIME = "torch"
DEFAULT_MODELS = ["gemma-2b-it:google/gemma-2b-it"]
DEFAULT_PRECISION = "bfloat16"
DEFAULT_SEQUENCE_LENGTH = 512


def _initialize_modeling_environment(
    dl_runtime: str,
    precision: str,
) -> None:
  """Configure the modeling environment."""
  if dl_runtime != DEFAULT_DL_RUNTIME:
    raise ValueError(
        f"The provided value `{dl_runtime}` for dl_runtime is not supported,"
        f" only `{DEFAULT_DL_RUNTIME}` is available."
    )

  # NOTE: Imported here and not at the top of the file to avoid
  # initialization issues with the environment variables above.
  import torch  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  torch.set_default_dtype(
      torch.bfloat16 if precision == "bfloat16" else torch.float32
  )


def get_models(
    models_config: Optional[Sequence[str]] = None,
    dl_runtime: str = DEFAULT_DL_RUNTIME,
    precision: str = DEFAULT_PRECISION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> lit_model.ModelMap:
  """Loads models from the given configs.

  Args:
    models_config: A list of model names and paths to load from, as
      "model:path", where path can be a URL, a local file path, or the name of
      a HuggingFace Transformers model.
    dl_runtime: The deep learning runtime that the model runs on, only "torch"
      is supported. All loaded models will use the same runtime,
      incompatibilities will result in errors.
    precision: Floating point precision for the models, either `bfloat16` or
      `float32`.
    batch_size: The number of examples to process per batch.
    max_length: The maximum sequence length of the input.

  Returns:
    A mapping from model name to initialized LIT model.
  """

  if not models_config:
    return {}

  # NOTE: Always call this function before initializing models to ensure the
  # environment is properly configured.
  _initialize_modeling_environment(dl_runtime, precision)

  from lit_nlp.examples.prompt_debugging import transformers_lms  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  models: dict[str, lit_model.Model] = {}
  for model_string in models_config:
    # Only split on the first ':' as path may be a URL containing 'https://'
    model_name, path = model_string.split(":", 1)
    logging.info("Loading model '%s' from '%s'", model_name, path)

    models |= transformers_lms.initialize_model_group_for_salience(
        model_name,
        model_name_or_path=path,
        batch_size=batch_size,
        framework=dl_runtime,
        max_length=max_length,
    )

  return models


def get_model_loaders(
    dl_runtime: str = DEFAULT_DL_RUNTIME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> lit_app.ModelLoadersMap:
  """Get the model loader for the configured runtime.

  Args:
    dl_runtime: The deep learning runtime that the model runs on, only "torch"
      is supported. All models are loaded with the same runtime,
      `model_name_or_path` incompatibilities will result in errors.
    batch_size: The default batch size.
    max_length: The default maximum sequence length.

  Returns:
    A mapping from model name to initialized LIT model.
  """

  from lit_nlp.examples.prompt_debugging import transformers_lms  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  transformers_init_spec: lit_types.Spec = {
      "model_name_or_path": lit_types.String(),
      "batch_size": lit_types.Integer(
          default=batch_size, min_val=1, max_val=64, required=False
      ),
      "max_length": lit_types.Integer(
          default=max_length, min_val=1, max_val=2048, required=False
      ),
      "framework": lit_types.CategoryLabel(
          vocab=transformers_lms.SUPPORTED_ML_RUNTIMES, default=dl_runtime
      ),
  }

  return {
      "Transformers LLM": (
          transformers_lms.initialize_model_group_for_salience,
          transformers_init_spec,
      )
  }
