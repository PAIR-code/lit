"""🐧 Penguin tabular dataset from the palmerpenguins project.

See https://allisonhorst.github.io/palmerpenguins/ for details.
"""

from typing import Optional

import pandas as pd

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.lib import file_cache

PENGUINS_URL = (
    'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/'
    'inst/extdata/penguins.csv'
)

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

  From https://allisonhorst.github.io/palmerpenguins/.
  """

  def __init__(self, max_examples: Optional[int] = None):
    path = file_cache.cached_path(PENGUINS_URL)
    dataset_df = pd.read_csv(path)
    # Match the field names used by the LIT spec.
    dataset_df = dataset_df.rename(columns={
        'bill_length_mm': 'culmen_length_mm',
        'bill_depth_mm': 'culmen_depth_mm',
    })
    dataset_df['sex'] = dataset_df['sex'].str.capitalize()

    # Filter out rows with missing values.
    fields = list(self.spec().keys())
    dataset_df = dataset_df.dropna(subset=fields)
    records = dataset_df.to_dict(orient='records')
    self._examples = [
        {field: rec[field] for field in fields} for rec in records
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
