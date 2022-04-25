# Copyright 2022 Google LLC
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
"""Module names and type definitions for frontend UI layouts."""
import enum
from typing import Any, Dict, List, Mapping, Optional, Text, Union

import attr
from lit_nlp.api import dtypes

JsonDict = Dict[Text, Any]


# LINT.IfChange
# pylint: disable=invalid-name
@enum.unique
class LitModuleName(dtypes.EnumSerializableAsValues, enum.Enum):
  """List of available frontend modules.

  Entries should map the TypeScript class name to the HTML element name,
  as declared in HTMLElementTagNameMap in the .ts file defining each LitModule.
  """
  AnnotatedTextModule = 'annotated-text-module'
  AnnotatedTextGoldModule = 'annotated-text-gold-module'
  AttentionModule = 'attention-module'
  ClassificationModule = 'classification-module'
  ColorModule = 'color-module'
  ConfusionMatrixModule = 'confusion-matrix-module'
  CounterfactualExplainerModule = 'counterfactual-explainer-module'
  CurvesModule = 'curves-module'
  DataTableModule = 'data-table-module'
  SimpleDataTableModule = 'simple-data-table-module'
  DatapointEditorModule = 'datapoint-editor-module'
  SimpleDatapointEditorModule = 'simple-datapoint-editor-module'
  DocumentationModule = 'documentation-module'
  EmbeddingsModule = 'embeddings-module'
  FeatureAttributionModule = 'feature-attribution-module'
  GeneratedImageModule = 'generated-image-module'
  GeneratedTextModule = 'generated-text-module'
  GeneratorModule = 'generator-module'
  LanguageModelPredictionModule = 'lm-prediction-module'
  MetricsModule = 'metrics-module'
  MultilabelModule = 'multilabel-module'
  PdpModule = 'pdp-module'
  RegressionModule = 'regression-module'
  SalienceClusteringModule = 'salience-clustering-module'
  SalienceMapModule = 'salience-map-module'
  ScalarModule = 'scalar-module'
  SequenceSalienceModule = 'sequence-salience-module'
  SliceModule = 'lit-slice-module'
  SpanGraphGoldModule = 'span-graph-gold-module'
  SpanGraphModule = 'span-graph-module'
  SpanGraphGoldModuleVertical = 'span-graph-gold-module-vertical'
  SpanGraphModuleVertical = 'span-graph-module-vertical'
  TCAVModule = 'tcav-module'
  TrainingDataAttributionModule = 'tda-module'
  ThresholderModule = 'thresholder-module'


# Most users should use LitModuleName, but we allow fallback to strings
# so that users can reference custom modules which are defined in TypeScript
# but not included in the LitModuleName enum above.
# If a string is used, it should be the HTML element name, like foo-bar-module.
LitModuleList = List[Union[str, LitModuleName]]


@attr.s(auto_attribs=True)
class LayoutSettings(dtypes.DataTuple):
  hideToolbar: bool = False
  mainHeight: int = 45
  centerPage: bool = False


@attr.s(auto_attribs=True)
class LitComponentLayout(dtypes.DataTuple):
  """Frontend UI layout (legacy); should match client/lib/types.ts."""
  # Keys are names of tabs; one must be called "Main".
  # Values are names of LitModule HTML elements,
  # e.g. data-table-module for the DataTableModule class.
  components: Dict[str, LitModuleList]
  layoutSettings: LayoutSettings = attr.ib(factory=LayoutSettings)
  description: Optional[str] = None

  def to_json(self) -> JsonDict:
    """Override serialization to properly convert nested objects."""
    # Not invertible, but these only go from server -> frontend anyway.
    return attr.asdict(self, recurse=True)


@attr.s(auto_attribs=True)
class LitCanonicalLayout(dtypes.DataTuple):
  """Frontend UI layout; should match client/lib/types.ts."""
  # Keys are names of tabs, and values are names of LitModule HTML elements,
  # e.g. data-table-module for the DataTableModule class.
  upper: Dict[str, LitModuleList]
  lower: Dict[str, LitModuleList] = attr.ib(factory=dict)
  layoutSettings: LayoutSettings = attr.ib(factory=LayoutSettings)
  description: Optional[str] = None

  def to_json(self) -> JsonDict:
    """Override serialization to properly convert nested objects."""
    # Not invertible, but these only go from server -> frontend anyway.
    return attr.asdict(self, recurse=True)


# TODO(b/205853382): remove dtypes.LitComponentLayout once references are
# updated.
LitComponentLayouts = Mapping[str, Union[LitComponentLayout, LitCanonicalLayout,
                                         dtypes.LitComponentLayout]]

# pylint: enable=invalid-name
# LINT.ThenChange(../client/lib/types.ts)
