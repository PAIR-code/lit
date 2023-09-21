"""🐧 Penguin tabular dataset from TFDS.

See https://www.tensorflow.org/datasets/catalog/penguins. for details.
"""

from collections.abc import Mapping
from typing import Optional, Union
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import tensorflow_datasets as tfds

VOCABS = {
    'island': ['Biscoe', 'Dream', 'Torgersen'],
    'sex': ['Female', 'Male'],
    'species': ['Adelie', 'Chinstrap', 'Gentoo']
}

INPUT_SPEC: lit_types.Spec = {
    'body_mass_g': lit_types.Scalar(min_val=2700, max_val=6300),
    'culmen_depth_mm': lit_types.Scalar(min_val=13, max_val=22),
    'culmen_length_mm': lit_types.Scalar(min_val=32, max_val=60),
    'flipper_length_mm': lit_types.Scalar(min_val=172, max_val=231),
    'island': lit_types.CategoryLabel(vocab=VOCABS['island']),
    'sex': lit_types.CategoryLabel(vocab=VOCABS['sex']),
}


class PenguinDataset(lit_dataset.Dataset):
  """Dataset of penguin tabular data.

  From https://www.tensorflow.org/datasets/catalog/penguins.
  """

  @classmethod
  def lit_example_from_record(cls, rec: Mapping[str, Union[float, int]]):
    return {
        'body_mass_g': rec['body_mass_g'],
        'culmen_depth_mm': rec['culmen_depth_mm'],
        'culmen_length_mm': rec['culmen_length_mm'],
        'flipper_length_mm': rec['flipper_length_mm'],
        'island': VOCABS['island'][rec['island']],
        'sex': VOCABS['sex'][rec['sex']],
        'species': VOCABS['species'][rec['species']],
    }

  def __init__(self, max_examples: Optional[int] = None):
    peng = tfds.load('penguins/simple', download=True, try_gcs=True)
    dataset_df = tfds.as_dataframe(peng['train'])

    # Filter out invalid rows.
    dataset_df = dataset_df.loc[dataset_df['sex'] != 2]
    records = dataset_df.to_dict(orient='records')
    self._examples = [
        PenguinDataset.lit_example_from_record(rec) for rec in records
    ][:max_examples]

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        'max_examples': lit_types.Integer(
            default=1000, min_val=0, max_val=10_000, required=False
        ),
    }

  def spec(self):
    return INPUT_SPEC | {
        'species': lit_types.CategoryLabel(vocab=VOCABS['species'])
    }
