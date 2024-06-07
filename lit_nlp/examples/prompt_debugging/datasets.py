"""Methods for configuring prompt debugging datasets."""

from collections.abc import Mapping, Sequence
import functools
import re
from typing import Optional

from absl import logging
from lit_nlp import app as lit_app
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.examples.datasets import lm as lm_data


DEFAULT_DATASETS = ["sample_prompts"]
DEFAULT_MAX_EXAMPLES = 1000

_plaintext_prompts = functools.partial(  # pylint: disable=invalid-name
    lm_data.PlaintextSents, field_name="prompt"
)
# Hack: normally dataset loaders are a class object which has a __name__,
# rather than a functools.partial
_plaintext_prompts.__name__ = "PlaintextSents"


def get_datasets(
    datasets_config: Optional[Sequence[str]] = None,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> Mapping[str, lit_dataset.Dataset]:
  """Loads datasets from the provided configs.

  Args:
    datasets_config: A sequence of configs in the form of <name>:<path> where
      the path points to is either: 1) a JSON Lines file containing records with
      a required "prompt" field and optional "target" and "source" fields; or 2)
      a plain text file where each line is a prompt.
    max_examples: Maximum number of examples in each loaded dataset.

  Returns:
    A mapping from dataset name to the initialized LIT dataset.
  """

  if not datasets_config:
    return {}

  datasets: dict[str, lit_dataset.Dataset] = {}
  for dataset_string in datasets_config:
    if dataset_string == "sample_prompts":
      dataset_name = "sample_prompts"
      path = lm_data.PromptExamples.SAMPLE_DATA_PATH
    else:
      # Only split on the first ':', because path may be a URL
      # containing 'https://'
      dataset_name, path = dataset_string.split(":", 1)
    logging.info("Loading dataset '%s' from '%s'", dataset_name, path)

    if path.endswith(".jsonl"):
      datasets[dataset_name] = lm_data.PromptExamples(path)
    # .txt or .txt-#####-of-#####
    elif path.endswith(".txt") or re.match(r".*\.txt-\d{5}-of-\d{5}$", path):
      datasets[dataset_name] = _plaintext_prompts(path)
    else:
      raise ValueError(f"Unsupported dataset format for {dataset_string}")

  for name in datasets:
    datasets[name] = datasets[name].slice[:max_examples]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))

  return datasets


def get_dataset_loaders() -> lit_app.DatasetLoadersMap:
  return {
      "jsonl_examples": (
          lm_data.PromptExamples,
          lm_data.PromptExamples.init_spec(),
      ),
      "plaintext_inputs": (
          _plaintext_prompts,
          lm_data.PlaintextSents.init_spec(),
      ),
  }
