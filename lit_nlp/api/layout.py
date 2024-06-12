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
from collections.abc import Mapping, Sequence
import enum
from typing import Optional, Union

import attr
from lit_nlp.api import dtypes


# pylint: disable=invalid-name
@enum.unique
class LitModuleName(dtypes.EnumSerializableAsValues, enum.Enum):
  """List of available frontend modules.

  Entries should map the TypeScript class name to the HTML element name,
  as declared in HTMLElementTagNameMap in the .ts file defining each LitModule.
  """
  # keep-sorted start
  AttentionModule = 'attention-module'
  ClassificationModule = 'classification-module'
  ConfusionMatrixModule = 'confusion-matrix-module'
  CurvesModule = 'curves-module'
  DataTableModule = 'data-table-module'
  DatapointEditorModule = 'datapoint-editor-module'
  DiveModule = 'dive-module'
  EmbeddingsModule = 'embeddings-module'
  FeatureAttributionModule = 'feature-attribution-module'
  GeneratedTextModule = 'generated-text-module'
  GeneratorModule = 'generator-module'
  LegacySequenceSalienceModule = 'legacy-sequence-salience-module'
  MetricsModule = 'metrics-module'
  MultilabelModule = 'multilabel-module'
  PdpModule = 'pdp-module'
  SalienceMapModule = 'salience-map-module'
  ScalarModule = 'scalar-module'
  SequenceSalienceModule = 'sequence-salience-module'
  SimpleDataTableModule = 'simple-data-table-module'
  # Simplified, non-replicating version of Datapoint Editor
  SimpleDatapointEditorModule = 'simple-datapoint-editor-module'
  # Non-replicating version of Datapoint Editor
  SingleDatapointEditorModule = 'single-datapoint-editor-module'
  ThresholderModule = 'thresholder-module'
  TrainingDataAttributionModule = 'tda-module'
  # keep-sorted end

  def __call__(self, **kw):
    return ModuleConfig(self.value, **kw)


# LINT.IfChange
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
LitModuleList = Sequence[Union[str, LitModuleName, ModuleConfig]]
# Keys are names of tabs, and values are names of LitModule HTML elements, e.g.,
# data-table-module for the DataTableModule class.
LitTabGroupLayout = Mapping[str, LitModuleList]


@attr.s(auto_attribs=True)
class LayoutSettings(dtypes.DataTuple):
  hideToolbar: bool = False
  mainHeight: int = 45
  leftWidth: int = 50
  centerPage: bool = False


@attr.s(auto_attribs=True)
class LitCanonicalLayout(dtypes.DataTuple):
  """Frontend UI layout; should match client/lib/types.ts."""
  upper: LitTabGroupLayout
  lower: LitTabGroupLayout = attr.ib(factory=dict)
  left: LitTabGroupLayout = attr.ib(factory=dict)
  layoutSettings: LayoutSettings = attr.ib(factory=LayoutSettings)
  description: Optional[str] = None

  def to_json(self) -> dtypes.JsonDict:
    """Override serialization to properly convert nested objects."""
    # Not invertible, but these only go from server -> frontend anyway.
    return attr.asdict(self, recurse=True)


LitComponentLayouts = Mapping[str, LitCanonicalLayout]

# pylint: enable=invalid-name
# LINT.ThenChange(../client/lib/types.ts)

##
# Common layout definitions.

modules = LitModuleName  # pylint: disable=invalid-name

MODEL_PREDS_MODULES = (
    modules.ClassificationModule,
    modules.MultilabelModule,
    modules.GeneratedTextModule,
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
            modules.LegacySequenceSalienceModule(requiredForTab=True),
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
        'on the page rather than being full width.'
    ),
)

THREE_PANEL_LAYOUT = LitCanonicalLayout(
    left={
        'Tabular Exploration': [modules.DataTableModule],
        'Current Example': [modules.DatapointEditorModule],
        'Visual Exploration': [modules.DiveModule],
        'Embeddings': [modules.EmbeddingsModule],
    },
    upper={
        'Predictions': MODEL_PREDS_MODULES,
        'Current Example': [modules.DatapointEditorModule],
        'Counterfactuals': [modules.GeneratorModule],
    },
    lower={
        'Metrics': [
            modules.MetricsModule,
            modules.ConfusionMatrixModule,
            modules.ThresholderModule,
        ],
        'Charts': [
            modules.ScalarModule,
            modules.PdpModule,
            modules.CurvesModule,
        ],
        'Explanations': [
            modules.SalienceMapModule,
            modules.LegacySequenceSalienceModule,
            modules.AttentionModule,
            modules.FeatureAttributionModule,
        ],
        'Influence': [modules.TrainingDataAttributionModule],
    },
    description=(
        'A three-panel layout with tools for exploring data in the aggregate or'
        ' per-example (on the left) or reviewing prediction results (upper'
        ' right) and performance characteristics, etc. (lower left).'
    ),
)

##
# A "kitchen sink" layout with maximum functionality.
STANDARD_LAYOUT = LitCanonicalLayout(
    upper={
        'Main': [
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
            modules.LegacySequenceSalienceModule,
            modules.AttentionModule,
            modules.FeatureAttributionModule,
        ],
        'Metrics': [
            modules.MetricsModule,
            modules.ConfusionMatrixModule,
            modules.CurvesModule,
            modules.ThresholderModule,
        ],
        'Influence': [modules.TrainingDataAttributionModule],
        'Counterfactuals': [
            modules.GeneratorModule,
        ],
    },
    description=(
        'The default LIT layout, which includes the data table and data point '
        'editor, the performance and metrics, predictions, explanations, and '
        'counterfactuals.'
    ),
)

DEFAULT_LAYOUTS = {
    'simple': SIMPLE_LAYOUT,
    'default': STANDARD_LAYOUT,
    'three_panel': THREE_PANEL_LAYOUT,
}
