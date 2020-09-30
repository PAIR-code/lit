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
import {LitModuleType} from './core/lit_module';
import {AttentionModule} from './modules/attention_module';
import {ClassificationModule} from './modules/classification_module';
import {ConfusionMatrixModule} from './modules/confusion_matrix_module';
import {CounterfactualExplainerModule} from './modules/counterfactual_explainer_module';
import {DataTableModule} from './modules/data_table_module';
import {DatapointEditorModule} from './modules/datapoint_editor_module';
import {EmbeddingsModule} from './modules/embeddings_module';
import {GeneratedTextModule} from './modules/generated_text_module';
import {GeneratorModule} from './modules/generator_module';
import {LanguageModelPredictionModule} from './modules/lm_prediction_module';
import {MetricsModule} from './modules/metrics_module';
import {PerturbationsTableModule} from './modules/perturbations_table_module';
import {PredictionScoreModule} from './modules/prediction_score_module';
import {RegressionModule} from './modules/regression_module';
import {SalienceMapModule} from './modules/salience_map_module';
import {SpanGraphGoldModule, SpanGraphModule} from './modules/span_graph_module';
import {LitComponentLayouts} from './services/modules_service';

// clang-format off
const MODEL_PREDS_MODULES: LitModuleType[] = [
  SpanGraphGoldModule,
  SpanGraphModule,
  ClassificationModule,
  RegressionModule,
  LanguageModelPredictionModule,
  GeneratedTextModule,
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
        ...MODEL_PREDS_MODULES
      ],
      'Data': [DataTableModule],
    },
    layoutSettings: {
      hideToolbar: true,
      mainHeight: 90,
      centerPage: true
    }
  },
  /**
   * A "simple demo server" layout for classifier models.
   * Assumes no metrics, embeddings, or attention.
   */
  'classifier':  {
    components : {
      'Main': [DataTableModule, DatapointEditorModule],
      'Classifiers': [
        ConfusionMatrixModule,
      ],
      'Counterfactuals': [GeneratorModule],
      'Predictions': [
        PredictionScoreModule,
        ...MODEL_PREDS_MODULES,
      ],
      'Explanations': [
        SalienceMapModule,
      ]
    }
  },
  /**
   * For masked language models
   */
  'lm':  {
    components : {
      'Main': [EmbeddingsModule, DataTableModule, DatapointEditorModule],
      'Predictions': [
        LanguageModelPredictionModule,
        ConfusionMatrixModule,
      ],
      'Counterfactuals': [GeneratorModule],
    }
  },
  /**
   * Simplified view for tagging/parsing models
   */
  'spangraph':  {
    components : {
      'Main': [DataTableModule, DatapointEditorModule],
      'Predictions': [
        SpanGraphGoldModule,
        SpanGraphModule,
      ],
      'Performance': [
        MetricsModule,
      ],
      'Counterfactuals': [GeneratorModule],
    }
  },
  /**
   * A default layout for LIT Modules
   */
  'default':  {
    components : {
      'Main': [EmbeddingsModule, DataTableModule, DatapointEditorModule],
      'Performance': [
        MetricsModule,
        ConfusionMatrixModule,
      ],
      'Predictions': [
        ...MODEL_PREDS_MODULES,
        PredictionScoreModule,

      ],
      'Explanations': [
        ...MODEL_PREDS_MODULES,
        SalienceMapModule,
        AttentionModule,
      ],
      'Counterfactuals': [GeneratorModule],
      'Counterfactual Explanation': [CounterfactualExplainerModule],
      'Counterfactual Changes': [
        PerturbationsTableModule
      ]
    }
  },
};
// clang-format on
