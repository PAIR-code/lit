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
"""Annotator implementations."""

from typing import List, Optional
import attr

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types
from lit_nlp.lib import utils


JsonDict = types.JsonDict
Spec = types.Spec


class PerFieldAnnotator(lit_components.Annotator):
  """Per-field annotator.

  Annotates individual fields from a dataset given an annotator model that takes
  a single input field in its input_spec() to create annotations.
  Each annotated field will be named as
  '{annotator name}:{annotator model output key}:{dataset field key}'.
  """

  def annotate(self, inputs: List[JsonDict],
               dataset: lit_dataset.Dataset,
               dataset_spec_to_annotate: Optional[types.Spec] = None):
    if len(self._annotator_model.input_spec().items()) != 1:
      raise ValueError('Annotator model provided to PerFieldAnnotator does not '
                       'operate on a single field')

    datasets = {}
    for input_name, input_type in self._annotator_model.input_spec().items():
      # Do remap of inputs based on input name needed by annotator.
      ds_keys = utils.find_spec_keys(dataset.spec(), type(input_type))
      for ds_key in ds_keys:
        temp_ds = lit_dataset.Dataset(examples=inputs, base=dataset)
        datasets[ds_key] = temp_ds.remap({ds_key: input_name})

    for ds_key, ds in datasets.items():
      outputs = self._annotator_model.predict(ds.examples)
      for output_name, output_type in self._annotator_model.output_spec(
          ).items():
        # Update dataset spec with new annotated field.
        field_name = f'{self._name}:{output_name}:{ds_key}'
        if dataset_spec_to_annotate:
          dataset_spec_to_annotate[field_name] = attr.evolve(
              output_type, annotated=True)

        # Update all examples with annotator output.
        for example, output in zip(inputs, outputs):
          example[field_name] = output[output_name]
