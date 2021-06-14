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
import '../elements/spinner';
import '../elements/tcav_score_bar';
import '@material/mwc-switch';

import {customElement, html} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {CallConfig, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {SliceService} from '../services/services';
import {STARRED_SLICE_NAME} from '../services/slice_service';

import {styles as sharedStyles} from './shared_styles.css';
import {styles} from './tcav_module.css';

const MIN_EXAMPLES_LENGTH = 3;  // minimum examples needed to train the CAV.
const TCAV_INTERPRETER_NAME = 'tcav';
const COLUMN_NAMES =
    ['Positive Slice', 'Run', 'Embedding', 'Class', 'CAV Score', 'Score Bar'];
const WARNING_TEXT =
    `this run was not statistically significant (p > 0.05)`;

/**
 * The TCAV module.
 */
@customElement('tcav-module')
export class TCAVModule extends LitModule {
  static get styles() {
    return [sharedStyles, styles];
  }
  static title = 'TCAV Explorer';
  static numCols = 12;
  static duplicateForModelComparison = true;

  static template = (model = '') => {
    return html`
      <tcav-module model=${model}>
      </tcav-module>`;
  };
  private readonly sliceService = app.getService(SliceService);

  @observable private readonly selectedSlices = new Set<string>();
  @observable private readonly selectedLayers = new Set<string>();
  @observable private readonly selectedClasses = new Set<string>();
  @observable private isLoading: boolean = false;
  @observable private isSliceHidden: boolean = false;
  @observable private isClassHidden: boolean = true;
  @observable private isEmbeddingHidden: boolean = true;

  private resultsTableData: TableData[] = [];
  private cavCounter = 0;

  @computed
  get modelSpec() {
    return this.appState.getModelSpec(this.model);
  }

  @computed
  get gradKeys() {
    return findSpecKeys(this.modelSpec.output, 'Gradients');
  }

  @computed
  get TCAVSliceNames() {
    return this.sliceService.sliceNames.filter(
        name => name !== STARRED_SLICE_NAME);
  }

  @computed
  get predClasses() {
    const predKeys = findSpecKeys(this.modelSpec.output, 'MulticlassPreds');
    // TODO(lit-dev): Handle the multi-headed case with more than one pred key.
    return this.modelSpec.output[predKeys[0]].vocab!;
  }

  @computed
  get nullIndex() {
    const predKeys = findSpecKeys(this.modelSpec.output, 'MulticlassPreds');
    // TODO(lit-dev): Handle the multi-headed case with more than one pred key.
    return this.modelSpec.output[predKeys[0]].null_idx!;
  }

  firstUpdated() {
    // Set the first grad key as default in selector.
    if (this.selectedLayers.size === 0 && this.gradKeys.length > 0) {
      this.selectedLayers.add(this.gradKeys[0]);
    }
    // Set first non-null pred class as default in selector.
    if (this.selectedClasses.size === 0 && this.predClasses.length > 1) {
      const initialIndex = this.nullIndex === 0 ? 1 : 0;
      this.selectedClasses.add(this.predClasses[initialIndex]);
    }
  }

  renderSpinner() {
    return html`
      <div class="spinner-container">
        <lit-spinner size=${24} color="var(--app-secondary-color)">
        </lit-spinner>
      </div>
    `;
  }

  renderCollapseBar(
      title: string, toggle: () => void, isHidden: boolean, items: string[],
      columnName: string, selectSet: Set<string>) {
    const checkboxChanged = (e: Event, item: string) => {
      const checkbox = e.target as HTMLInputElement;
      if (checkbox.checked) {
        selectSet.add(item);
      } else {
        selectSet.delete(item);
      }
    };
    const data = items.map((item) => {
      const row = [
        // clang-format off
        html`<lit-checkbox ?checked=${selectSet.has(item)}
                @change='${(e: Event) => {checkboxChanged(e, item);}}'>
            </lit-checkbox>`,
        // clang-format on
        item
      ];
      return row;
    });
    const columns = ['selected', columnName];

    // clang-format off
    return html`
      <div class='collapse-bar' @click=${toggle}>
        <div class="axis-title">
          <div>${title}</div>
        </div>
        <mwc-icon class="icon-button min-button">
          ${isHidden ? 'expand_more' : 'expand_less'}
        </mwc-icon>
      </div>
      <div class='collapse-content'
        style=${styleMap({'display': `${isHidden ? 'none' : 'block'}`})}>
        ${data.length === 0 ?
          html`<div class='require-text label-2'>
                  This module requires at least one ${columnName.toLowerCase()}.
               </div>` :
          html`<lit-data-table .data=${data} .columnNames=${columns}>
               </lit-data-table>`}
      </div>
    `;
    // clang-format on
  }

  render() {
    const shouldDisable = () => {
      for (const slice of this.selectedSlices) {
        const examples = this.sliceService.getSliceByName(slice);
        if (examples == null) return true;
        const comparisonSetLength =
            this.appState.currentInputData.length - examples.length;
        if (examples.length >= MIN_EXAMPLES_LENGTH &&
            comparisonSetLength >= MIN_EXAMPLES_LENGTH) {
          return false;  // only enable if slice has minimum number of examples
        }
      }
      return true;
    };

    const clearOptions = () => {
      this.selectedClasses.clear();
      this.selectedLayers.clear();
      this.selectedSlices.clear();
    };

    const clearTable = () => {
      this.resultsTableData = [];
      this.cavCounter = 0;
      this.requestUpdate();
    };

    const toggleSliceCollapse = () => {
      this.isSliceHidden = !this.isSliceHidden;
    };
    const toggleClassCollapse = () => {
      this.isClassHidden = !this.isClassHidden;
    };
    const toggleEmbeddingCollapse = () => {
      this.isEmbeddingHidden = !this.isEmbeddingHidden;
    };

    const cavCount = this.selectedClasses.size * this.selectedSlices.size *
        this.selectedLayers.size;

    const disabledText =
        `select a slice with ${MIN_EXAMPLES_LENGTH} or more examples`;

    // The width of the SVG increase by 60px for each additional entry after
    // the first bar, so their labels don't overlap.
    // clang-format off
    // TODO(lit-dev): Switch the current barchart viz to a table-based viz.
    return html`
      <div class="outer-container">
        <div class="left-container">
          <div class="controls-holder">
            ${this.renderCollapseBar('Select Slices',
                                     toggleSliceCollapse,
                                     this.isSliceHidden,
                                     this.TCAVSliceNames,
                                     'Slice',
                                     this.selectedSlices)}
            ${this.renderCollapseBar('Explainable Classes',
                                     toggleClassCollapse,
                                     this.isClassHidden,
                                     this.predClasses,
                                     'Class',
                                     this.selectedClasses)}
            ${this.renderCollapseBar('Embeddings',
                                     toggleEmbeddingCollapse,
                                     this.isEmbeddingHidden,
                                     this.gradKeys,
                                     'Embedding',
                                     this.selectedLayers)}
          </div>
          <div class="controls-actions">
            <div id='examples-selected-label'
              class="label-2">${cavCount} CAV Results</div>
            <div class="controls-buttons">
              <button id='clear-button' class='text-button'
                @click=${clearOptions}
                ?disabled=${this.selectedClasses.size === 0 &&
                  this.selectedLayers.size === 0 &&
                  this.selectedSlices.size === 0}>Clear</button>
              <button id='submit'
                class='text-button' title=${shouldDisable() ? disabledText: ''}
                @click=${() => this.runTCAV()} ?disabled=${
                 shouldDisable()}>Run TCAV</button>
            </div>
          </div>
        </div>
        <div id='vis-container' class=${classMap({'loading': this.isLoading})}>
          ${this.isLoading ? this.renderSpinner(): ''}
          <div class="output-controls">
            <button class='clear-table-button text-button' @click=${clearTable}
               ?disabled=${this.resultsTableData.length === 0}>Clear</button>
          </div>
          <lit-data-table
              .columnNames=${COLUMN_NAMES}
              .data=${[...this.resultsTableData]}
          ></lit-data-table>
        </div>
      </div>
    `;
    // clang-format on
  }

  private async runSingleTCAV(config: CallConfig, sliceName: string) {
    const comparisonSetLength = this.appState.currentInputData.length -
        config['concept_set_ids'].length;
    if (config['concept_set_ids'].length < MIN_EXAMPLES_LENGTH ||
        comparisonSetLength < MIN_EXAMPLES_LENGTH) {
      return;
    }

    // All indexed inputs in the dataset are passed in, with the concept set
    // ids specified in the config.
    // TODO(b/178210779): Enable caching in the component's predict call.
    const result = await this.apiService.getInterpretations(
        this.appState.currentInputData, this.model,
        this.appState.currentDataset, TCAV_INTERPRETER_NAME, config,
        `Running ${TCAV_INTERPRETER_NAME}`);

    if (result === null) {
      return;
    }
    // TODO(lit-dev): Show local TCAV scores in the scalar chart.
    const score = result[0]['result']['score'];
    return {
      'slice': sliceName,
      'config': config,
      'score': result[0]['p_val'] < 0.05 ? score : '-',
      'random_mean': result[0]['random_mean']
    };
  }

  private async runTCAV() {
    this.isLoading = true;

    // TODO(lit-dev): Add option to run TCAV on selected examples.
    // TODO(lit-dev): Add option to run TCAV on categorical features.
    const promises = [];
    for (const slice of this.selectedSlices.values()) {
      for (const gradClass of this.selectedClasses.values()) {
        for (const layer of this.selectedLayers.values()) {
          const selectedIds = this.sliceService.getSliceByName(slice)!;
          const config: CallConfig = {
            'concept_set_ids': selectedIds,
            'class_to_explain': gradClass,
            'grad_layer': layer,
            'dataset_name': this.appState.currentDataset,
          };
          promises.push(this.runSingleTCAV(config, slice));
        }
      }
    }

    const results = await this.loadLatest('getResults', Promise.all(promises));
    if (results == null) return;

    for (const res of results) {
      if (res == null) continue;
      if (res['config'] == null || res['score'] == null) continue;
      this.resultsTableData.push({
        'Positive Slice': res['slice'],
        'Run': this.cavCounter,
        'Embedding': res['config']['grad_layer'],
        'Class': res['config']['class_to_explain'],
        'CAV Score': res['score'],
        // clang-format off
        'Score Bar': res['score'] === '-' ? WARNING_TEXT :
            html`<tcav-score-bar
                   score=${res['score']}
                   meanVal=${res['random_mean']}
                   clampVal=${1}>
                 </tcav-score-bar>`
        // clang-format on
      });
    }
    this.cavCounter++;
    this.isLoading = false;
    this.requestUpdate();
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    const supportsEmbs = doesOutputSpecContain(modelSpecs, 'Embeddings');
    const supportsGrads = doesOutputSpecContain(modelSpecs, 'Gradients');
    const multiclassPreds =
        doesOutputSpecContain(modelSpecs, 'MulticlassPreds');
    if (supportsGrads && supportsEmbs && multiclassPreds) {
      return true;
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'tcav-module': TCAVModule;
  }
}
