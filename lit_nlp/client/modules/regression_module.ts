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
import {customElement, html} from 'lit-element';
import {observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {IndexedInput, ModelsMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {RegressionService} from '../services/services';

import {styles} from './regression_module.css';
import {styles as sharedStyles} from './shared_styles.css';

interface RegressionResult {
  [key: string]: number;
}

interface ResultElement {
  'header': string;
  'result': string;
}

/**
 * A LIT module that renders regression results.
 */
@customElement('regression-module')
export class RegressionModule extends LitModule {
  static title = 'Regression Results';
  static duplicateForExampleComparison = true;
  static numCols = 2;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<regression-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></regression-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly regressionService = app.getService(RegressionService);

  @observable private results: RegressionResult[] = [];

  firstUpdated() {
    const getPrimarySelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(
        getPrimarySelectedInputData, primarySelectedInputData => {
          this.updateSelection(primarySelectedInputData);
        });
  }

  private async updateSelection(primarySelectedInputData: IndexedInput|null) {
    this.results = [];
    if (primarySelectedInputData == null) {
      return;
    }

    const selectedInputData = [primarySelectedInputData];

    const dataset = this.appState.currentDataset;
    const promise = this.regressionService.getRegressionPreds(
        selectedInputData, this.model, dataset);

    const results = await this.loadLatest('regressionPreds', promise);
    if (results === null) return;

    if (results.length > 0) {
      const keys = Object.keys(results[0]);
      for (let i = 0; i < selectedInputData.length; i++) {
        for (const key of keys) {
          const regressionInfo = (await this.regressionService.getResults(
              [selectedInputData[i].id], this.model, key))[0];
          results[i][this.regressionService.getErrorKey(key)] =
              regressionInfo.error;
        }
      }
    }
    this.results = results;
  }

  render() {
    const primarySelectedInputData =
        this.selectionService.primarySelectedInputData;
    if (primarySelectedInputData == null) {
      return null;
    }
    // We only show the primary selection, so input is always one example.
    const input = primarySelectedInputData;

    // Use the spec to find which fields we should display.
    const spec = this.appState.getModelSpec(this.model);
    const scoreFields: string[] = findSpecKeys(spec.output, 'RegressionScore');


    const rows: ResultElement[][] = [];
    // Display parent field, score and error on the same row per output.
    for (const scoreField of scoreFields) {
      // Add new row for each output from the model.
      const row = [] as ResultElement[];
      const score =
          (this.results.length === 0 || this.results[0][scoreField] == null) ?
          '' :
          this.results[0][scoreField].toFixed(4);
      if (score) {
        row.push({'header': scoreField, 'result': score} as ResultElement);
      }
      // Target score to compare against.
      const parentField = spec.output[scoreField].parent! || '';
      const parentScore = input.data[parentField] == null ?
          '' :
          input.data[parentField].toFixed(4);
      if (parentField && parentScore) {
        row.push(
            {'header': parentField, 'result': parentScore} as ResultElement);
      }
      const errorScore =
          (this.results.length === 0 ||
           this.results[0][this.regressionService.getErrorKey(scoreField)] ==
               null) ?
          '' :
          this.results[0][this.regressionService.getErrorKey(scoreField)]
              .toFixed(4);
      if (errorScore) {
        row.push({'header': 'error', 'result': errorScore} as ResultElement);
      }
      rows.push(row);
    }

    const renderRow = (row: ResultElement[]) => html`
      <tr>
        ${row.map((entry) => html`<th>${entry.header}</th>`)}
      </tr>
      <tr>
        ${row.map((entry) => html`<td>${entry.result}</td>`)}
      </tr>`;

    return html`
        <table>
          ${rows.map((row) => renderRow(row))}
        </table>`;
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'RegressionScore');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'regression-module': RegressionModule;
  }
}
