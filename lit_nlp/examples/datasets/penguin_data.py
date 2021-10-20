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
    filtered_pd = dataset_df.loc[dataset_df['sex'] != 2]

    examples = filtered_pd.to_dict(orient='records')

    # Convert categorical features to strings.
    for ex in examples:
      ex['island'] = VOCABS['island'][ex['island']]
      ex['sex'] = VOCABS['sex'][ex['sex']]
      ex['species'] = VOCABS['species'][ex['species']]

    self._examples = examples

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
