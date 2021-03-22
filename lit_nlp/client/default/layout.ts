/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Import Modules
import '../modules/span_graph_module';
import {LitModuleType} from '../core/lit_module';
import {LitComponentLayouts} from '../lib/types';
import {AnnotatedTextGoldModule, AnnotatedTextModule} from '../modules/annotated_text_module';
import {AttentionModule} from '../modules/attention_module';
import {ClassificationModule} from '../modules/classification_module';
import {ColorModule} from '../modules/color_module';
import {ConfusionMatrixModule} from '../modules/confusion_matrix_module';
import {CounterfactualExplainerModule} from '../modules/counterfactual_explainer_module';
import {DataTableModule} from '../modules/data_table_module';
import {DatapointEditorModule} from '../modules/datapoint_editor_module';
import {EmbeddingsModule} from '../modules/embeddings_module';
import {GeneratedTextModule} from '../modules/generated_text_module';
import {GeneratorModule} from '../modules/generator_module';
import {LanguageModelPredictionModule} from '../modules/lm_prediction_module';
import {MetricsModule} from '../modules/metrics_module';
import {RegressionModule} from '../modules/regression_module';
import {SalienceMapModule} from '../modules/salience_map_module';
import {ScalarModule} from '../modules/scalar_module';
import {SliceModule} from '../modules/slice_module';
import {SpanGraphGoldModuleVertical, SpanGraphModuleVertical} from '../modules/span_graph_module';
import {TCAVModule} from '../modules/tcav_module';

// clang-format off
const MODEL_PREDS_MODULES: LitModuleType[] = [
  SpanGraphGoldModuleVertical,
  SpanGraphModuleVertical,
  ClassificationModule,
  RegressionModule,
  LanguageModelPredictionModule,
  GeneratedTextModule,
  AnnotatedTextGoldModule,
  AnnotatedTextModule,
];
// clang-format on

// clang-format off
/**
 * Possible layouts for LIT (component groups and settigns.)
 */
export const LAYOUTS: LitComponentLayouts = {
  /**
   * A "simple demo server" layout optimized for single models.
   */
  'simple':  {
    components : {
      'Main': [
        DatapointEditorModule,
      ],
      'Predictions': [ ...MODEL_PREDS_MODULES],
      'Data': [DataTableModule],
    },
    layoutSettings: {
      hideToolbar: true,
      mainHeight: 30,
      centerPage: true
    },
    description: 'A basic layout just containing a datapoint creator/editor, the predictions, and the data table. There are also some visual simplifications: the toolbar is hidden, and the modules are centered on the page rather than being full width.'
  },
  /**
   * A "simple demo server" layout for classifier models.
   * Assumes no metrics, embeddings, or attention.
   */
  'classifier':  {
    components : {
      'Main': [DataTableModule, DatapointEditorModule, SliceModule,
               ColorModule],
      'Classifiers': [
        ConfusionMatrixModule,
      ],
      'Counterfactuals': [GeneratorModule],
      'Predictions': [
        ScalarModule,
        ...MODEL_PREDS_MODULES,
      ],
      'Explanations': [
        SalienceMapModule,
      ]
    },
    description: "A default layout for classification results, which shows the data table and datapoint editor, as well as the predictions and counterfactuals."
  },
  /**
   * For masked language models
   */
  'lm':  {
    components : {
      'Main': [EmbeddingsModule, DataTableModule, DatapointEditorModule,
               SliceModule, ColorModule],
      'Predictions': [
        LanguageModelPredictionModule,
        ConfusionMatrixModule,
      ],
      'Counterfactuals': [GeneratorModule],
    },
    description: "A layout optimized for language modeling, which includes the language modeling and confusion matrix modules, as well as the standard the embedding projector, data table, datapoint module, and counterfactuals."
  },
  /**
   * Simplified view for tagging/parsing models
   */
  'spangraph':  {
    components : {
      'Main': [DataTableModule, DatapointEditorModule, SliceModule,
               ColorModule],
      'Predictions': [
        SpanGraphGoldModuleVertical,
        SpanGraphModuleVertical,
      ],
      'Performance': [
        MetricsModule,
      ],
      'Counterfactuals': [GeneratorModule],
    },
    description: "A layout optimized for span graph prediction, which includes the span graph module, as well as the standard data table, datapoint module, and counterfactuals."
  },
  /**
   * A default layout for LIT Modules without EmbeddingsModule
   * TODO(lit-dev): move to a custom frontend build,
   * or remove this if b/159186274 is resolved to speed up page load.
   */
  'default_no_projector':  {
    components : {
      'Main': [DataTableModule, DatapointEditorModule, SliceModule,
               ColorModule],
      'Performance': [
        MetricsModule,
        ConfusionMatrixModule,
        TCAVModule,
      ],
      'Predictions': [
        ...MODEL_PREDS_MODULES,
        ScalarModule,
      ],
      'Explanations': [
        ...MODEL_PREDS_MODULES,
        SalienceMapModule,
        AttentionModule,
      ],
      'Counterfactuals': [GeneratorModule, CounterfactualExplainerModule],
    },
    description: "A default LIT layout, which includes the data table and data point editor, the performance and metrics, predictions, explanations, and counterfactuals. Does not include the embedding projector."
  },
  /**
   * A default layout for LIT Modules
   */
  'default':  {
    components : {
      'Main': [EmbeddingsModule, DataTableModule, DatapointEditorModule,
               SliceModule, ColorModule],
      'Performance': [
        MetricsModule,
        ConfusionMatrixModule,
        TCAVModule,
      ],
      'Predictions': [
        ...MODEL_PREDS_MODULES,
        ScalarModule,
      ],
      'Explanations': [
        ...MODEL_PREDS_MODULES,
        SalienceMapModule,
        AttentionModule,
      ],
      'Counterfactuals': [GeneratorModule, CounterfactualExplainerModule],
    },
    description: "The default LIT layout, which includes the data table and data point editor, the performance and metrics, predictions, explanations, and counterfactuals."
  },
};
// clang-format on
