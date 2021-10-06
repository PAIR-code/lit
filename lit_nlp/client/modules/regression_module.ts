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
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {RegressionService} from '../services/services';

import {styles} from './regression_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

interface RegressionResult {
  [key: string]: number;
}

/**
 * A LIT module that renders regression results.
 */
@customElement('regression-module')
export class RegressionModule extends LitModule {
  static override title = 'Regression Results';
  static override duplicateForExampleComparison = true;
  static override numCols = 3;
  static override template = (model = '', selectionServiceIndex = 0) => {
    return html`<regression-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></regression-module>`;
  };

  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly regressionService = app.getService(RegressionService);

  @observable private result: RegressionResult|null = null;

  override firstUpdated() {
    const getPrimarySelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(
        getPrimarySelectedInputData, primarySelectedInputData => {
          this.updateSelection(primarySelectedInputData);
        });
  }

  private async updateSelection(inputData: IndexedInput|null) {
    if (inputData == null) {
      this.result = null;
      return;
    }

    const dataset = this.appState.currentDataset;
    const promise = this.regressionService.getRegressionPreds(
        [inputData], this.model, dataset);

    const results = await this.loadLatest('regressionPreds', promise);
    if (results === null || results.length === 0) {
      this.result = null;
      return;
    }

    // Extract the single result, as this only is for a single input.
    const keys = Object.keys(results[0]);
    for (const key of keys) {
      const regressionInfo = (await this.regressionService.getResults(
          [inputData.id], this.model, key))[0];
      results[0][this.regressionService.getErrorKey(key)] =
          regressionInfo.error;
    }
    this.result = results[0];
  }

  override render() {
    if (this.result == null) {
      return null;
    }
    const result = this.result;
    const input = this.selectionService.primarySelectedInputData!;

    // Use the spec to find which fields we should display.
    const spec = this.appState.getModelSpec(this.model);
    const scoreFields: string[] = findSpecKeys(spec.output, 'RegressionScore');


    const rows: TableData[] = [];
    let hasParent = false;
    // Per output, display score, and parent field and error if available.
    for (const scoreField of scoreFields) {
      // Add new row for each output from the model.
      const score = result[scoreField] == null ?
          '' :
          result[scoreField].toFixed(4);
      // Target score to compare against.
      const parentField = spec.output[scoreField].parent! || '';
      const parentScore = input.data[parentField] == null ?
          '' :
          input.data[parentField].toFixed(4);
      let errorScore = '';
      if (parentField && parentScore) {
        const error =
            result[this.regressionService.getErrorKey(scoreField)];
        if (error != null) {
          hasParent = true;
          errorScore = error.toFixed(4);
        }
      }
      rows.push({
        'Field': scoreField,
        'Ground truth': parentScore,
        'Score': score,
        'Error': errorScore
      });
    }

    // If no fields have ground truth scores to compare then don't display the
    // ground truth-related columns.
    const columnNames = hasParent ?
        ['Field', 'Ground truth', 'Score', 'Error'] :
        ['Field', 'Score'];

    return html`
      <lit-data-table
        .columnNames=${columnNames}
        .data=${rows}
      ></lit-data-table>`;
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'RegressionScore');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'regression-module': RegressionModule;
  }
}
