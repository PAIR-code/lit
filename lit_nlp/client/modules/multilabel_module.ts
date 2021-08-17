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
import {customElement, html, TemplateResult} from 'lit-element';
import {styleMap} from 'lit-html/directives/style-map';
import {observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {formatBoolean, IndexedInput, ModelInfoMap, Preds, Spec} from '../lib/types';
import {doesOutputSpecContain} from '../lib/utils';

import {styles} from './multilabel_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

interface DisplayInfo {
  label: string;
  value: number;
  isGroundTruth?: boolean;
}

/** Model output module class. */
@customElement('multilabel-module')
export class MultilabelModule extends LitModule {
  static title = 'Multilabel Results';
  static duplicateForExampleComparison = true;
  static numCols = 3;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<multilabel-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></multilabel-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  @observable private labeledPredictions: {[name: string]: DisplayInfo[]} = {};

  firstUpdated() {
    const getSelectedInput = () =>
        this.selectionService.primarySelectedInputData;
    this.react(getSelectedInput, selectedInput => {
      this.updateSelection();
    });
    // Update once on init, to avoid duplicate calls.
    this.updateSelection();
  }

  private async updateSelection() {
    const data = this.selectionService.primarySelectedInputData;
    if (data === null) {
      this.labeledPredictions = {};
      return;
    }

    const result = this.apiService.getPreds(
        [data], this.model, this.appState.currentDataset,
        ['SparseMultilabelPreds']);
    const preds = await result;
    if (preds === null) return;

    this.labeledPredictions = await this.parseResult(preds[0], data);
  }

  private async parseResult(result: Preds, data: IndexedInput) {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const predictedKeys = Object.keys(result);
    const labeledPredictions: {[name: string]: DisplayInfo[]} = {};

    for (let predIndex = 0; predIndex < predictedKeys.length; predIndex++) {
      const predictionName = predictedKeys[predIndex];
      const labelField = outputSpec[predictionName].parent;
      const preds = result[predictionName] as Array<[string, number]>;
      const labeledExample = preds.map((pred: [string, number], i: number) => {
        const dict: DisplayInfo = {
          value: pred[1],
          label: pred[0],
        };
        if (labelField != null && data.data[labelField].indexOf(dict['label']) !== -1) {
          dict.isGroundTruth = true;
        }
        return dict;
      });
      labeledPredictions[predictionName] = labeledExample;
    }
    return labeledPredictions;
  }

  render() {
    const keys = Object.keys(this.labeledPredictions);
    return html`
        ${keys.map((key) => this.renderRow(key, this.labeledPredictions[key]))}
    `;
  }

  private renderBar(fieldName: string, pred: DisplayInfo): TemplateResult {
    // TODO(b/181692911): Style through CSS when data table supports it.
    const pad = 0.75;
    const margin = 0.35;
    const barStyle: {[name: string]: string} = {};
    const scale = 90;
    barStyle['width'] = `${scale * pred['value']}%`;
    barStyle['background-color'] = '#07a3ba';
    barStyle['padding-left'] = `${pad}%`;
    barStyle['padding-right'] = `${pad}%`;
    barStyle['margin-left'] = `${margin}%`;
    barStyle['margin-right'] = `${margin}%`;
    const holderStyle: {[name: string]: string} = {};
    holderStyle['width'] = '100px';
    holderStyle['height'] = '20px';
    holderStyle['display'] = 'flex';
    holderStyle['position'] = 'relative';
    return html`
        <div style='${styleMap(holderStyle)}'>
          <div style='${styleMap(barStyle)}'></div>
        </div>`;
  }

  private renderRow(fieldName: string, prediction: DisplayInfo[]) {
    const rows: TableData[] = prediction.map((pred) => {
      const row = [
        pred['label'],
        formatBoolean(pred['isGroundTruth']!),
        (+pred['value']).toFixed(3),
        this.renderBar(fieldName, pred)
      ];
      return row;
    });
    const columnNames = ["Class", "Label", "Score", "Score Bar"];

    return html`
        <div class='classification-row-holder'>
          <div class='classification-row-title'>${fieldName}</div>
          <lit-data-table
            .columnNames=${columnNames}
            .data=${rows}
          ></lit-data-table>
        </div>`;
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'SparseMultilabelPreds');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'multilabel-module': MultilabelModule;
  }
}
