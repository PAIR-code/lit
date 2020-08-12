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
import {css, customElement, html, property} from 'lit-element';
import {observable, reaction} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {IndexedInput, ModelsMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {RegressionService} from '../services/services';

import {styles as sharedStyles} from './shared_styles.css';

interface RegressionResult {
  [key: string]: number;
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
    return [
      sharedStyles,
      css`
        td {
          padding-right: 4px;
        }
      `,
    ];
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
    const inputs = [primarySelectedInputData];


    // Use the spec to find which fields we should display.
    const spec = this.appState.getModelSpec(this.model);
    const textFields: string[] = findSpecKeys(spec.input, 'TextSegment');
    const scoreFields: string[] = findSpecKeys(spec.output, 'RegressionScore');

    const headers: string[] = [];
    const rows: Array<Array<string|number>> = [];
    inputs.forEach((input) => {
      rows.push([] as Array<string|number>);
    });
    // Add columns for score fields and the parent label fields.
    for (const scoreField of scoreFields) {
      // Target score to compare against.
      const parentField = spec.output[scoreField].parent! || '-';
      headers.push(parentField);
      headers.push(scoreField);
      headers.push('error');
      rows.forEach((row, i) => {
        row.push(
            inputs[i].data[parentField] == null ?
                '-' :
                inputs[i].data[parentField].toFixed(4));
        row.push(
            (this.results.length === 0 || this.results[i][scoreField] == null) ?
                '-' :
                this.results[i][scoreField].toFixed(4));
        row.push(
            (this.results.length === 0 ||
             this.results[i][this.regressionService.getErrorKey(scoreField)] ==
                 null) ?
                '-' :
                this.results[i][this.regressionService.getErrorKey(scoreField)]
                    .toFixed(4));
      });
    }
    if (inputs.length > 1) {
      // Add columns for text fields, so we can see which example each score
      // corresponds to.
      // TODO(lit-team): clip text if too long, or there are many input fields.
      for (const textField of textFields) {
        headers.push(textField);
        rows.forEach((row, i) => {
          row.push(inputs[i].data[textField]);
        });
      }
    }

    const renderRow = (row: Array<string|number>) => html`
      <tr>
        ${row.map((entry) => html`<td>${entry}</td>`)}
      </tr>`;

    return html`
        <table>
          <tr>
            ${headers.map((header) => html`<th>${header}</th>`)}
          </tr>
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
