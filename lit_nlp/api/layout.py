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
  ConfusionMatrixModule = 'confusion-matrix-module'
  CounterfactualExplainerModule = 'counterfactual-explainer-module'
  CurvesModule = 'curves-module'
  DataTableModule = 'data-table-module'
  SimpleDataTableModule = 'simple-data-table-module'
  DatapointEditorModule = 'datapoint-editor-module'
  SimpleDatapointEditorModule = 'simple-datapoint-editor-module'
  DiveModule = 'dive-module'
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
  SpanGraphGoldModule = 'span-graph-gold-module'
  SpanGraphModule = 'span-graph-module'
  SpanGraphGoldModuleVertical = 'span-graph-gold-module-vertical'
  SpanGraphModuleVertical = 'span-graph-module-vertical'
  TCAVModule = 'tcav-module'
  TrainingDataAttributionModule = 'tda-module'
  ThresholderModule = 'thresholder-module'

  def __call__(self, **kw):
    return ModuleConfig(self.value, **kw)


# TODO(lit-dev): consider making modules subclass this instead of LitModuleName.
@attr.s(auto_attribs=True)
class ModuleConfig(dtypes.DataTuple):
  module: Union[str, LitModuleName]
  requiredForTab: bool = False
  # TODO(b/172979677): support title, duplicateAsRow, numCols,
  # and startMinimized.


# Most users should use LitModuleName, but we allow fallback to strings
# so that users can reference custom modules which are defined in TypeScript
# but not included in the LitModuleName enum above.
# If a string is used, it should be the HTML element name, like foo-bar-module.
LitModuleList = List[Union[str, LitModuleName, ModuleConfig]]


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


LitComponentLayouts = Mapping[str, Union[LitComponentLayout,
                                         LitCanonicalLayout]]

# pylint: enable=invalid-name
# LINT.ThenChange(../client/lib/types.ts)

##
# Common layout definitions.

modules = LitModuleName  # pylint: disable=invalid-name

MODEL_PREDS_MODULES = (
    modules.SpanGraphGoldModuleVertical,
    modules.SpanGraphModuleVertical,
    modules.ClassificationModule,
    modules.MultilabelModule,
    modules.RegressionModule,
    modules.LanguageModelPredictionModule,
    modules.GeneratedTextModule,
    modules.AnnotatedTextGoldModule,
    modules.AnnotatedTextModule,
    modules.GeneratedImageModule,
)

DEFAULT_MAIN_GROUP = (
    modules.DataTableModule,
    modules.DatapointEditorModule,
)

##
# A "simple demo server" layout.
SIMPLE_LAYOUT = LitCanonicalLayout(
    upper={
        'Editor': [
            modules.DocumentationModule,
            modules.SimpleDatapointEditorModule,
        ],
        'Examples': [modules.SimpleDataTableModule],
    },
    lower={
        'Predictions': list(MODEL_PREDS_MODULES),
        'Salience': [
            *MODEL_PREDS_MODULES,
            modules.SalienceMapModule(requiredForTab=True),
        ],
        'Sequence Salience': [
            *MODEL_PREDS_MODULES,
            modules.SequenceSalienceModule(requiredForTab=True),
        ],
        'Influence': [modules.TrainingDataAttributionModule],
    },
    layoutSettings=LayoutSettings(
        hideToolbar=True,
        mainHeight=30,
        centerPage=True,
    ),
    description=(
        'A basic layout just containing a datapoint creator/editor, the '
        'predictions, and the data table. There are also some visual '
        'simplifications: the toolbar is hidden, and the modules are centered '
        'on the page rather than being full width.'),
)

##
# A "kitchen sink" layout with maximum functionality.
STANDARD_LAYOUT = LitCanonicalLayout(
    upper={
        'Main': [
            modules.DocumentationModule,
            modules.EmbeddingsModule,
            *DEFAULT_MAIN_GROUP,
        ]
    },
    lower={
        'Predictions': [
            *MODEL_PREDS_MODULES,
            modules.ScalarModule,
            modules.PdpModule,
        ],
        'Explanations': [
            *MODEL_PREDS_MODULES,
            modules.SalienceMapModule,
            modules.SequenceSalienceModule,
            modules.AttentionModule,
            modules.FeatureAttributionModule,
        ],
        'Salience Clustering': [modules.SalienceClusteringModule],
        'Metrics': [
            modules.MetricsModule,
            modules.ConfusionMatrixModule,
            modules.CurvesModule,
            modules.ThresholderModule,
        ],
        'Influence': [modules.TrainingDataAttributionModule],
        'Counterfactuals': [
            modules.GeneratorModule,
            modules.CounterfactualExplainerModule,
        ],
        'TCAV': [modules.TCAVModule],
    },
    description=(
        'The default LIT layout, which includes the data table and data point '
        'editor, the performance and metrics, predictions, explanations, and '
        'counterfactuals.'),
)

DEFAULT_LAYOUTS = {
    'simple': SIMPLE_LAYOUT,
    'default': STANDARD_LAYOUT,
}
