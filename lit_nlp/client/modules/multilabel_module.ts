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
import '../elements/score_bar';
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {SortableTemplateResult, TableData, TableEntry} from '../elements/table';
import {SparseMultilabelPreds} from '../lib/lit_types';
import {formatBoolean, IndexedInput, ModelInfoMap, NumericResults, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {SelectionService} from '../services/services';

import {styles} from './multilabel_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

// Dict of prediction results for each predicted class.
interface PredKeyResultInfo {
  [className: string]: NumericResults;
}

// Contains all prediction results for display by the module.
interface AllResultsInfo {
  [predKey: string]: PredKeyResultInfo;
}


/** Model output module class. */
@customElement('multilabel-module')
export class MultilabelModule extends LitModule {
  static override title = 'Multilabel Results';
  static override duplicateForModelComparison = false;
  static override numCols = 3;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <multilabel-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </multilabel-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  @observable private resultsInfo: AllResultsInfo = {};
  private datapoints: IndexedInput[] = [];
  private readonly groundTruthLabels = new Set<string>();
  private maxValues: {[predKey: string]: number} = {};

  override firstUpdated() {
    const getSelectedInput = () =>
        this.selectionService.primarySelectedInputData;
    this.react(getSelectedInput, selectedInput => {
      this.updateSelection();
    });
    this.react(() => this.appState.compareExamplesEnabled,
        compareExamplesEnabled => {
      this.updateSelection();
    });
    // Update once on init, to avoid duplicate calls.
    this.updateSelection();
  }

  private async updateSelection() {
    const data = this.selectionService.primarySelectedInputData;
    if (data === null) {
      this.resultsInfo = {};
      return;
    }

    // Collect datapoints to predict.
    const selectionServices = app.getServiceArray(SelectionService);
    const datapoints: IndexedInput[] = [];
    for (const selectionService of selectionServices) {
      const selected = selectionService.primarySelectedInputData;
      if (selected != null) {
        datapoints.push(selected);
      }
      if (!this.appState.compareExamplesEnabled) {
        break;
      }
    }
    this.datapoints = datapoints;

    // Run predictions on all models.
    const models = this.appState.currentModels;
    const results = await Promise.all(models.map(
        async model => this.apiService.getPreds(
            datapoints, model, this.appState.currentDataset,
            [SparseMultilabelPreds])));
    if (results === null) {
      this.resultsInfo = {};
      return;
    }

    // Store all ground truth labels for all models, for use in display.
    this.groundTruthLabels.clear();
    for (const model of models) {
      const outputSpec = this.appState.currentModelSpecs[model].spec.output;
      const predKeys = findSpecKeys(outputSpec, SparseMultilabelPreds);
      for (const predKey of predKeys) {
        const labelField =
            (outputSpec[predKey] as SparseMultilabelPreds).parent;
        if (labelField != null) {
          this.groundTruthLabels.add(labelField);
        }
      }
    }

    // Parse results for display purposes.
    this.maxValues = {};
    const allResults: AllResultsInfo = {};
    for (let modelIdx = 0; modelIdx < results.length; modelIdx++) {
      for (let datapointIdx = 0; datapointIdx < results[modelIdx].length;
           datapointIdx++) {
        for (const predKey of Object.keys(results[modelIdx][datapointIdx])) {
          if (allResults[predKey] == null) {
            allResults[predKey] = {};
          }
          const preds = results[modelIdx][datapointIdx][predKey];
          for (const labelAndScore of preds) {
            const label = labelAndScore[0];
            const score = labelAndScore[1];
            if (allResults[predKey][label] == null) {
              allResults[predKey][label] = {};
            }
            let key = `${models[modelIdx]}`;
            // If there are two datapoints, then label them.
            if (results[modelIdx].length > 1) {
              key += (datapointIdx === 0) ? ' selected' : ' pinned';
            }
            allResults[predKey][label][key] = score;
            if (this.maxValues[predKey] == null) {
              this.maxValues[predKey] = 0;
            }
            if (score > this.maxValues[predKey]) {
              this.maxValues[predKey] = score;
            }
          }
        }
      }
    }
    this.resultsInfo = allResults;
  }

  override renderImpl() {
    const keys = Object.keys(this.resultsInfo);
    return html`
        <div class="top-holder">
          ${keys.map((key) => this.renderTable(key, this.resultsInfo[key]))}
        </div>
    `;
  }

  private renderBar(val: number, maxScore: number): SortableTemplateResult {
    return {
      template: html`
          <score-bar score=${val} maxScore=${maxScore}></score-bar>`,
      value: val
    };
  }

  private renderTable(fieldName: string, prediction: PredKeyResultInfo) {
    const columnNames = [`${fieldName} class`];

    // Add columns for ground truth labels.
    for (let i = 0; i < this.datapoints.length; i++) {
      for (const label of this.groundTruthLabels) {
        let labelColName = label;
        if (this.datapoints.length > 1) {
          labelColName += i === 0 ? ' selected' : ' pinned';
        }
        columnNames.push(labelColName);
      }
    }

    // Add columns for model predictions and deltas of predictions against
    // the first model/datapoint prediction.
    const cols = new Set<string>();
    for (const row of Object.values(prediction)) {
      for (const key of Object.keys(row)) {
       cols.add(key);
      }
    }
    const predCols = Array.from(cols.values());
    for (let i = 0; i < predCols.length; i++) {
      const col = predCols[i];
      columnNames.push(col);
      if (i > 0) {
        columnNames.push(col + ' Δ');
      }
    }

    const maxScore = Math.ceil(this.maxValues[fieldName]);
    // Create the table data.
    const rows: TableData[] = Object.keys(prediction).map((className) => {
      const row: TableEntry[] = [className];
      for (let i = 0; i < this.datapoints.length; i++) {
        const ex = this.datapoints[i];
        for (const label of this.groundTruthLabels.keys()) {
          // TODO(lit-dev): consider making a Set() for faster membership
          // checking if we have a large # of labels.
          row.push(formatBoolean(ex.data[label].includes(className)));
        }
      }
      let baselineScore = 0;
      for (let i = 0; i < predCols.length; i++) {
        const col = predCols[i];
        const score = prediction[className][col] == null ? 0 :
            prediction[className][col];
        row.push(this.renderBar(score, maxScore));
        if (i === 0) {
          baselineScore = score;
        } else {
          const delta = score - baselineScore;
          row.push(delta.toFixed(3));
        }
      }
      return row;
    });

    return html`
        <div class="table-holder">
          <lit-data-table
            .columnNames=${columnNames}
            .data=${rows}
            searchEnabled
          ></lit-data-table>
        </div>`;
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, SparseMultilabelPreds);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'multilabel-module': MultilabelModule;
  }
}
