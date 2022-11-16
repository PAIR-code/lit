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

// tslint:disable:no-new-decorators
import '@material/mwc-switch';
import '../elements/score_bar';

import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {ColumnHeader, SortableTemplateResult, TableData} from '../elements/table';
import {MulticlassPreds} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {CalculatedColumnType} from '../services/data_service';
import {DataService, SelectionService} from '../services/services';

import {styles} from './classification_module.css';

const SPARSE_MODE_THRESHOLD = 0.01;

interface DisplayInfo {
  value: number;
  isTruth: boolean;
  isPredicted: boolean;
}

interface LabelRows {
  [label: string]: DisplayInfo[];
}

interface LabeledPredictions {
  [predKey: string]: LabelRows;
}

/** Model output module class. */
@customElement('classification-module')
export class ClassificationModule extends LitModule {
  static override title = 'Classification Results';
  static override referenceURL =
      'https://github.com/PAIR-code/lit/wiki/components.md#classification';
  static override duplicateForExampleComparison = false;
  static override duplicateForModelComparison = false;
  static override numCols = 3;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => {
        return html`
      <classification-module model=${model} .shouldReact=${shouldReact}
        selectionServiceIndex=${selectionServiceIndex}>
      </classification-module>`;
      };

  static override get styles() {
    return [sharedStyles, styles];
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap,
                                      datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, MulticlassPreds);
  }

  private readonly dataService = app.getService(DataService);
  private readonly pinnedSelectionService =
      app.getService(SelectionService, 'pinned');

  @observable private sparseMode = false;
  @observable private labeledPredictions: LabeledPredictions = {};

  override firstUpdated() {
    const getSelectionChanges = () => [
      this.appState.compareExamplesEnabled, this.appState.currentModels,
      this.pinnedSelectionService.primarySelectedInputData, this.sparseMode,
      this.selectionService.primarySelectedInputData, this.dataService.dataVals
    ];
    this.reactImmediately(getSelectionChanges, () => {this.updateSelection();});
  }

  private async updateSelection() {
    const data: IndexedInput[] = [];

    // If we're in comparison mode, always put the pinned datapoint first
    if (this.appState.compareExamplesEnabled &&
        this.pinnedSelectionService.primarySelectedInputData) {
      data.push(this.pinnedSelectionService.primarySelectedInputData);
    }

    // Add any selected datapoint
    if (this.selectionService.primarySelectedInputData) {
      data.push(this.selectionService.primarySelectedInputData);
    }

    // If no pinned or selected datapoint, bail
    if (data.length === 0) {
      this.labeledPredictions = {};
      return;
    }

    // Create an expansion panel for each <model, predicition head> pair
    // TODO(ryanmullins) Adjust logic so that models with the same output_spec
    // are grouped into the same expansion panel, using the prediciton head as
    // the panel label.
    for (const model of this.appState.currentModels) {
      const labeledPredictions = this.parseResult(model, data);
      Object.assign(this.labeledPredictions, labeledPredictions);
    }
  }

  /**
   * Creates a LabeledPredictions object that is displayed as a series of tables
   * inside expansion panels. The keys of this object are the prediction heads
   * and the values are dictionaries with a key for each class in the vocabulary
   * and arrays of DisplayInfo values for the pinned and selected datapoints.
   */
  private parseResult(model: string, inputs: IndexedInput[]):
      LabeledPredictions {
    // currentModelSpecs getter accesses appState.metadata.models before init???
    const {output} = this.appState.currentModelSpecs[model].spec;
    const multiclassKeys = findSpecKeys(output, MulticlassPreds);
    const labeledPredictions: LabeledPredictions = {};

    // Iterate over the multiclass prediction heads
    for (const predKey of multiclassKeys) {
      const topLevelKey = this.dataService.getColumnName(model, predKey);
      const predClassKey = this.dataService.getColumnName(
          model, predKey, CalculatedColumnType.PREDICTED_CLASS);
      labeledPredictions[topLevelKey] = {};
      const {parent, vocab} = output[predKey] as MulticlassPreds;
      const scores =
          inputs.map(input => this.dataService.getVal(input.id, topLevelKey));
      const predictedClasses =
          inputs.map(input => this.dataService.getVal(input.id, predClassKey));
      // If no vocab provided, create a list of strings of the class indices.
      const labels =
          vocab || Array.from({length: scores[0].length}, (v, k) => `${k}`);

      // Iterate over the vocabulary for this prediction head
      for (let i = 0; i < labels.length; i++) {
        const label = labels[i];

        // Map the predctions for each example into DisplayInfo objects
        const rowPreds = [];

        for (let j = 0; j < scores.length; j++) {
          const score = scores[j];

          // Only push null scores if not in sparseMode
          if (score == null) {
            if (!this.sparseMode) {
              rowPreds.push({value: 0, isPredicted: false, isTruth: false});
            }
            continue;
          }

          const value = score[i];
          const isPredicted = label === predictedClasses[j];
          const {data} = inputs[j];
          const isTruth = (parent != null && data[parent] === labels[i]);
          // Push values if not in sparseMode or if above threshold
          if (!this.sparseMode || value >= SPARSE_MODE_THRESHOLD) {
            rowPreds.push({value, isPredicted, isTruth});
          }
        }

        if (rowPreds.length) labeledPredictions[topLevelKey][label] = rowPreds;
      }
    }

    return labeledPredictions;
  }

  override renderImpl() {
    const clsFieldSpecs =
        this.appState.currentModels.flatMap((model) =>
            Object.values(this.appState.currentModelSpecs[model].spec.output)
                  .filter(
                    (fieldSpec) => fieldSpec instanceof MulticlassPreds
                  ) as MulticlassPreds[]);

    const hasGroundTruth = clsFieldSpecs.some((fs) =>
        fs.parent != null && fs.parent in this.appState.currentDatasetSpec);

    const allowSparseMode = clsFieldSpecs.some((fs) => fs.vocab.length > 10);

    const onClickSwitch = () => {this.sparseMode = !this.sparseMode;};

    // clang-format off
    return html`<div class='module-container'>
      <div class="module-results-area">
        ${
        Object.entries(this.labeledPredictions)
            .map(([fieldName, labelRow], i, arr) => {
              const featureTable =
                  this.renderFeatureTable(labelRow, hasGroundTruth);
              return arr.length === 1 ? featureTable : html`
                  <expansion-panel .label=${fieldName} expanded>
                    ${featureTable}
                  </expansion-panel>`;
            })}
      </div>
      <div class="module-footer">
        <annotated-score-bar-legend ?hasTruth=${hasGroundTruth}>
        </annotated-score-bar-legend>
        ${allowSparseMode ? html`
            <div class='switch-container' @click=${onClickSwitch}>
              <div>Only show classes above ${SPARSE_MODE_THRESHOLD}</div>
              <mwc-switch ?selected=${this.sparseMode}></mwc-switch>
            </div>` : null}
      </div>
    </div>`;
    // clang-format on
  }

  private renderFeatureTable(labelRow: LabelRows, hasGroundTruth: boolean) {
    function renderDisplayInfo(pred: DisplayInfo): SortableTemplateResult {
      return {
        template: html`<annotated-score-bar .value=${pred.value}
                                            ?isPredicted=${pred.isPredicted}
                                            ?isTruth=${pred.isTruth}
                                            ?hasTruth=${hasGroundTruth}>
        </annotated-score-bar>`,
        value: pred.value
      };
    }

    const rows = Object.entries(labelRow).map(([label, values]) => {
      const row: TableData = [label, ...values.map(renderDisplayInfo)];

      // values.length will be at most 2. In this case we have a pinned and
      // selected datapoint, and add the delta between their values to the row.
      if (values.length === 2) {
        const pinned = values[0].value;
        const selected = values[1].value;
        row.push(Math.abs(pinned - selected));
      }

      return row;
    });

    // If a row has more than two values, we have a pinned and selected
    // datapoint and the delta between them, so we need 4 column names.
    // Otherwise, we figure out the two column names given the value of
    // compareExamplesEnabled -- true = pinned, false = selected.
    const columnNames: Array<string|ColumnHeader> =
      rows[0].length > 2 ? [
        {name: 'Class', rightAlign: false},
        "Score - Pinned", "Score - Selected", "Δ(Pinned, Selected)"
      ] : this.appState.compareExamplesEnabled ? [
        {name: 'Class', rightAlign: false}, "Score - Pinned"
      ] : [{name: 'Class', rightAlign: false}, "Score"];

    return html`<lit-data-table .columnNames=${columnNames} .data=${rows}>
                </lit-data-table>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'classification-module': ClassificationModule;
  }
}
