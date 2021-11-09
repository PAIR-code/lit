"""🐧 Penguin tabular dataset from TFDS.

See https://www.tensorflow.org/datasets/catalog/penguins. for details.
"""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import tensorflow_datasets as tfds

VOCABS = {
    'island': ['Biscoe', 'Dream', 'Torgersen'],
    'sex': ['Female', 'Male'],
    'species': ['Adelie', 'Chinstrap', 'Gentoo']
}


class PenguinDataset(lit_dataset.Dataset):
  """Dataset of penguin tabular data.

  From https://www.tensorflow.org/datasets/catalog/penguins.
  """

  def __init__(self):
    peng = tfds.load('penguins/simple', download=True, try_gcs=True)
    dataset_df = tfds.as_dataframe(peng['train'])

    # Filter out invalid rows
    dataset_df = dataset_df.loc[dataset_df['sex'] != 2]

    records = dataset_df.to_dict(orient='records')
    for rec in records:
      ex = {
          'body_mass_g': rec['body_mass_g'],
          'culmen_depth_mm': rec['culmen_depth_mm'],
          'culmen_length_mm': rec['culmen_length_mm'],
          'flipper_length_mm': rec['flipper_length_mm'],
          'island': VOCABS['island'][rec['island']],
          'sex': VOCABS['sex'][rec['sex']],
          'species': VOCABS['species'][rec['species']]
      }
      self._examples.append(ex)

  def spec(self):
    return {
        'body_mass_g': lit_types.Scalar(),
        'culmen_depth_mm': lit_types.Scalar(),
        'culmen_length_mm': lit_types.Scalar(),
        'flipper_length_mm': lit_types.Scalar(),
        'island': lit_types.CategoryLabel(vocab=VOCABS['island']),
        'sex': lit_types.CategoryLabel(vocab=VOCABS['sex']),
        'species': lit_types.CategoryLabel(vocab=VOCABS['species']),
    }
