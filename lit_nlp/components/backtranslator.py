# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Lint as: python3
"""Backtranslation generator through Google Cloud Translate API."""
from typing import List, Sequence, Tuple, Text, Any, Optional

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import pandas as pd

from google.cloud import translate_v2 as translate

JsonDict = types.JsonDict

FIELDS_TO_BACKTRANSLATE_KEY = 'Fields to backtranslate'


class Backtranslator(lit_components.Generator):
  """Use Cloud Translate API as a Generator.

  In order to use this generator, you must have Cloud Translation set up
  through a Google Cloud project as described at
  https://cloud.google.com/translate/docs/setup and have downloaded your
  application credentials file locally and set the
  GOOGLE_APPLICATION_CREDENTIALS environment variable to point to that file.
  """

  def __init__(self,
               source_language: Text = 'en',
               pivot_languages: Sequence[Text] = ('fr', 'de')):
    self._source_lang = source_language
    self._pivot_langs = list(pivot_languages)
    self._translate_client = translate.Client()

  def config_spec(self) -> types.Spec:
    return {
        FIELDS_TO_BACKTRANSLATE_KEY:
            types.MultiFieldMatcher(
                spec='input',
                types=['TextSegment'],
                select_all=True),
    }

  def generate_all(self,
                   inputs: List[JsonDict],
                   model: lit_model.Model,
                   dataset: lit_dataset.Dataset,
                   config: Optional[JsonDict] = None) -> List[List[JsonDict]]:
    """Run generation on a set of inputs.

    If more than one field is to be backtranslated, each field is independently
    backtranslated per example. For example, if there are two fields to be
    backtranslated, this method will generate two examples per pivot language.

    Use this batch API by default, so we can make parallel requests.
    Args:
      inputs: sequence of inputs, following dataset.spec()
      model: (unused)
      dataset: dataset, used to access dataset.spec()
      config: additional runtime options

    Returns:
      list of list of new generated inputs, following dataset.spec()
    """
    outputs = self.run(inputs, dataset, config=config)
    return outputs

  def run(self,
          inputs: List[JsonDict],
          dataset: lit_dataset.Dataset,
          config: Optional[JsonDict] = None):
    """Run generation on a set of inputs.

    Args:
      inputs: sequence of inputs, following dataset.spec()
      dataset: dataset, used to access dataset.spec()
      config: additional runtime options

    Returns:
      list of list of new generated inputs, following dataset.spec()
    """
    all_outputs = [[] for _ in inputs]

    config = config or {}

    # Find text fields.
    text_fields = utils.find_spec_keys(dataset.spec(), types.TextSegment)
    # If config key is missing, backtranslate all text fields.
    fields_to_backtranslate = list(
        config.get(FIELDS_TO_BACKTRANSLATE_KEY, text_fields))
    candidates_by_field = {}
    for field_name in fields_to_backtranslate:
      texts = [ex[field_name] for ex in inputs]
      candidates_by_field[field_name] = self.generate_from_texts(texts)
    # Generate by substituting in each field.
    # TODO(lit-team): substitute on a combination of fields?
    for field_name in candidates_by_field:
      candidates = candidates_by_field[field_name]
      for i, ex in enumerate(inputs):
        for candidate in candidates[i]:
          new_ex = utils.copy_and_update(ex, {field_name: candidate})
          all_outputs[i].append(new_ex)
    return all_outputs

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Generate from a single example."""
    return self.generate_all([example], model, dataset, config=config)[0]

  def generate_from_texts(self,
                          texts: List[Text]) -> Tuple[List[List[Text]], Any]:
    """Run backtranslation on the list of strings."""
    # Use Pandas to keep track of metadata, so we can batch MT inputs
    # without losing track of which example they belong to.
    # Prepare input DataFrame
    dataframes = []
    for lang in self._pivot_langs:
      df = pd.DataFrame(data={'source': texts}).reset_index()
      df['pivot_language'] = lang
      dataframes.append(df)
    df = pd.concat(dataframes, axis=0).sort_values(by='index')
    # Forward translation
    # pylint: disable=g-complex-comprehension
    mt_inputs = [{
        'source': text,
        'source_language': self._source_lang,
        'target_language': lang
    } for text, lang in zip(df['source'], df['pivot_language'])]
    result = []
    for mt_input in mt_inputs:
      result.append(
          self._translate_client.translate(
              mt_input['source'],
              target_language=mt_input['target_language'],
              source_language=mt_input['source_language']))
    all_translations = [[r['translatedText']] for r in result]
    # Track metadata by replicating input rows
    # TODO(iftenney): replace with DataFrame.explode() once we can use
    # pandas 0.25
    rows = []
    for i, translation_set in enumerate(all_translations):
      for translation in translation_set:
        row = dict(df.iloc[i])
        row['pivot'] = translation
        rows.append(row)
    # Forward translation results
    intermediate_df = pd.DataFrame.from_records(rows)
    # TODO(lit-team): yield a chunk with intermediate state at this point,
    # for visualization before reverse translate is complete.
    # Reverse translation
    # pylint: disable=g-complex-comprehension
    mt_inputs = [{
        'source': text,
        'source_language': src,
        'target_language': self._source_lang
    } for text, src in zip(intermediate_df['pivot'],
                           intermediate_df['pivot_language'])]
    logging.info('Reverse: %d translations requested.', len(mt_inputs))
    result = []
    for mt_input in mt_inputs:
      result.append(
          self._translate_client.translate(
              mt_input['source'],
              target_language=mt_input['target_language'],
              source_language=mt_input['source_language']))
    all_translations = [[r['translatedText']] for r in result]
    # Track metadata by replicating input rows
    # TODO(iftenney): replace with DataFrame.explode() once we can use
    # pandas 0.25
    rows = []
    for i, translation_set in enumerate(all_translations):
      for translation in translation_set:
        row = dict(intermediate_df.iloc[i])
        row['target'] = translation
        rows.append(row)
    final_df = pd.DataFrame.from_records(rows)
    # Since we kept the indices in the DataFrame, we can group over these to get
    # the paraphrase candidates for each input.
    # this gives a list(list(str))
    candidates = list(
        final_df.groupby(by='index').agg({'target': list})['target'])
    return candidates
