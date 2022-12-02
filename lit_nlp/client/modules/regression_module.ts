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
import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {RegressionScore} from '../lib/lit_types';
import {IndexedInput, ModelInfoMap, RegressionResults, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {CalculatedColumnType} from '../services/data_service';
import {DataService} from '../services/services';

import {styles} from './regression_module.css';

/**
 * A LIT module that renders regression results.
 */
@customElement('regression-module')
export class RegressionModule extends LitModule {
  static override title = 'Regression Results';
  static override duplicateForExampleComparison = true;
  static override numCols = 3;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <regression-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </regression-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly dataService = app.getService(DataService);
  @observable private result: RegressionResults|null = null;

  override firstUpdated() {
    const getSelectionChanges = () =>
        [this.selectionService.primarySelectedInputData,
         this.dataService.dataVals];
    this.reactImmediately(getSelectionChanges, () => {
      this.updateSelection(this.selectionService.primarySelectedInputData);
    });
  }

  private async updateSelection(inputData: IndexedInput|null) {
    if (inputData == null) {
      this.result = null;
      return;
    }

    const result: RegressionResults = {};
    const {output} = this.appState.getModelSpec(this.model);
    const scoreFields: string[] = findSpecKeys(output, RegressionScore);
    for (const key of scoreFields) {
      const predKey = this.dataService.getColumnName(this.model, key);
      const errorKey = this.dataService.getColumnName(
          this.model, key, CalculatedColumnType.ERROR);
      result[key] = {
        score: this.dataService.getVal(inputData.id, predKey),
        error: this.dataService.getVal(inputData.id, errorKey)
      };
    }
    this.result = result;
  }

  override renderImpl() {
    if (this.result == null) {
      return null;
    }
    const result = this.result;
    const input = this.selectionService.primarySelectedInputData!;

    // Use the spec to find which fields we should display.
    const spec = this.appState.getModelSpec(this.model);
    const scoreFields: string[] = findSpecKeys(spec.output, RegressionScore);


    const rows: TableData[] = [];
    let hasParent = false;
    // Per output, display score, and parent field and error if available.
    for (const scoreField of scoreFields) {
      // Add new row for each output from the model.
      const score = result[scoreField]?.score?.toFixed(4) || '';
      // Target score to compare against.
      const parentField =
          (spec.output[scoreField] as RegressionScore).parent || '';
      const parentScore = input.data[parentField]?.toFixed(4) || '';
      let errorScore = '';
      if (parentField && parentScore) {
        const error = result[scoreField].error;
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
    return doesOutputSpecContain(modelSpecs, RegressionScore);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'regression-module': RegressionModule;
  }
}
