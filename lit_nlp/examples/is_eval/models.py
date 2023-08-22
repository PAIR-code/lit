"""Custom GLUE Model and ModelSpec for the Input Salience Evaluation demo."""
from typing import cast
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.examples.is_eval import datasets as is_eval_datasets
from lit_nlp.examples.models import glue_models


class ISEvalModel(glue_models.SST2Model):
  """Custom GLUE model for the Input Salience Evaluation demo."""

  def __init__(self, model_name: str, *args, **kw):
    """Initializes a custom SST-2 model for the Input Salience Eval demo.

    Args:
      model_name: The model's name. Used to determine dataset compatibility.
      *args: Additional positional args to pass to the SST2Model base class.
      **kw: Additional keyword args to pass to the SST2Model base class.
    """
    super().__init__(*args, **kw)
    self._model_name = model_name

  def is_compatible_with_dataset(self, dataset: lit_dataset.Dataset) -> bool:
    """Returns true if the model is compatible with the dataset.

    The Input Salience Eval demo is somewhat unique in that each model and
    dataset have compatible specs but the intention is to pair them for
    specific tasks.

    This class determines compatibility by:

    1.  Ensuring that the value of `model_name` is contained in the `default`
        value of the `dataset_name` field in the provided `dataset_spec`.
    2.  Calling super().is_compatible_with_dataset() to check compatibility
        using the base ModelSpec check.

    Args:
      dataset: The dataset for which compatibility will be determined.
    """
    if not isinstance(dataset,
                      is_eval_datasets.SingleInputClassificationFromTSV):
      return False

    eval_dataset = cast(is_eval_datasets.SingleInputClassificationFromTSV,
                        dataset)
    if self.model_name in eval_dataset.name:
      return super().is_compatible_with_dataset(dataset)
    else:
      return False
