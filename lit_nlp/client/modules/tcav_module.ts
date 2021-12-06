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

import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {TemplateResult} from 'lit';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {CallConfig, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {SliceService} from '../services/services';
import {STARRED_SLICE_NAME} from '../services/slice_service';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {styles} from './tcav_module.css';

const MIN_EXAMPLES_LENGTH = 3;  // minimum examples needed to train the CAV.
const TCAV_INTERPRETER_NAME = 'tcav';
const COLUMN_NAMES = [
  'Positive Slice', 'Negative Slice', 'Run', 'Embedding', 'Class', 'CAV Score',
  'Score Bar'
];

const RELATIVE_TCAV_MIN_EXAMPLES = 6;  // MIN_SPLIT_SIZE * MIN_SPLITS in tcav.py
const NO_T_TESTING_WARNING = `Did not run t-testing (requires at least ${
    RELATIVE_TCAV_MIN_EXAMPLES} positive and ${
    RELATIVE_TCAV_MIN_EXAMPLES} negative samples)`;

const MAX_P_VAL = 0.05;
const HIGH_P_VAL_WARNING =
    `this run was not statistically significant (p > ${MAX_P_VAL})`;

interface TcavResults {
  positiveSlice: string;
  negativeSlice: string;
  config: CallConfig;
  score: number;
  // tslint:disable-next-line:enforce-name-casing
  p_val: number;
  // tslint:disable-next-line:enforce-name-casing
  random_mean: number;
}

/**
 * The TCAV module.
 */
@customElement('tcav-module')
export class TCAVModule extends LitModule {
  static override get styles() {
    return [sharedStyles, styles];
  }
  static override title = 'TCAV Explorer';
  static override numCols = 12;
  static override duplicateForModelComparison = true;

  static override template = (model = '') => {
    return html`
      <tcav-module model=${model}>
      </tcav-module>`;
  };
  private readonly sliceService = app.getService(SliceService);

  @observable private readonly selectedSlices = new Set<string>();
  @observable private readonly selectedLayers = new Set<string>();
  @observable private readonly selectedClasses = new Set<string>();
  @observable private readonly negativeSlices = new Set<string>();
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

  // Returns pairs in the format [positive slice, negative slice (or null)]
  // for slices selected in the settings.
  @computed
  get slicePairs(): Array<[string, string|null]> {
    const positiveSlices: string[] = Array.from(this.selectedSlices.values());
    const negativeSlices: string[] = Array.from(this.negativeSlices.values());

    if (positiveSlices.length === 0) return [];

    if (negativeSlices.length === 0) {
      return positiveSlices.map((slice: string) => {
        return [slice, null];
      });
    }
    const pairs: Array<[string, string | null]> = [];
    for (const positiveSlice of positiveSlices) {
      for (const negativeSlice of negativeSlices) {
        pairs.push([positiveSlice, negativeSlice]);
      }
    }
    return pairs;
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

  override firstUpdated() {
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
      title: string, barToggled: () => void, isHidden: boolean, items: string[],
      columnName: string, selectSet: Set<string>, secondSelectName: string = '',
      secondSelectSet: Set<string>|null = null) {
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
      if (secondSelectSet != null) {
        const secondCheckboxChanged = (e: Event, item: string) => {
          const checkbox = e.target as HTMLInputElement;
          if (checkbox.checked) {
            secondSelectSet.add(item);
          } else {
            secondSelectSet.delete(item);
          }
        };
        row.push(
            // clang-format off
            html`<lit-checkbox id='compare-switch'
                  ?checked=${secondSelectSet.has(item)}
                  @change='${(e: Event) => {secondCheckboxChanged(e, item);}}'>
                </lit-checkbox>`
            // clang-format on
        );
      }
      return row;
    });
    const columns = ['selected', columnName];
    if (secondSelectSet != null) {
      columns.push(secondSelectName);
    }

    // clang-format off
    return html`
      <div class='collapse-bar' @click=${barToggled}>
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
          html`<lit-data-table .verticalAlignMiddle=${true}
                               .columnNames=${columns}
                               .data=${data}></lit-data-table>`}
      </div>
    `;
    // clang-format on
  }

  override render() {
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
      this.negativeSlices.clear();
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
      <div class="module-container">
        <div class="module-content">
          <div class="left-container">
            <div class="controls-holder">
              ${this.renderCollapseBar('Select Slices',
                                       toggleSliceCollapse,
                                       this.isSliceHidden,
                                       this.TCAVSliceNames,
                                       'Positive slice',
                                       this.selectedSlices,
                                       'Negative slice', this.negativeSlices)}
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
                <button id='clear-button' class="hairline-button"
                  @click=${clearOptions}
                  ?disabled=${this.selectedClasses.size === 0 &&
                    this.selectedLayers.size === 0 &&
                    this.selectedSlices.size === 0 &&
                    this.negativeSlices.size === 0}>Clear</button>
                <button id='submit'
                  class="hairline-button" title=${shouldDisable() ? disabledText: ''}
                  @click=${() => this.runTCAV()} ?disabled=${
                   shouldDisable()}>Run TCAV</button>
              </div>
            </div>
          </div>
          <div id='vis-container' class=${classMap({'loading': this.isLoading})}>
            ${this.isLoading ? this.renderSpinner(): ''}
            <lit-data-table .columnNames=${COLUMN_NAMES}
                            .data=${[...this.resultsTableData]}
                            .verticalAlignMiddle=${true}></lit-data-table>
          </div>
        </div>
        <div class="module-footer">
          <button class="hairline-button" @click=${clearTable}
             ?disabled=${this.resultsTableData.length === 0}>Clear</button>
        </div>
      </div>
    `;
    // clang-format on
  }

  private async runSingleTCAV(
      config: CallConfig, positiveSlice: string, negativeSlice: string):
      Promise<TcavResults|undefined> {
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
    return {
      'positiveSlice': positiveSlice,
      'negativeSlice': negativeSlice,
      'config': config,
      'score': result[0]['result']['score'],
      'p_val': result[0]['p_val'],
      'random_mean': result[0]['random_mean']
    };
  }

  private async runTCAV() {
    this.isLoading = true;

    // TODO(lit-dev): Add option to run TCAV on selected examples.
    // TODO(lit-dev): Add option to run TCAV on categorical features.
    const promises: Array<Promise<TcavResults|undefined>> = [];
    for (const slicePair of this.slicePairs) {
      for (const gradClass of this.selectedClasses.values()) {
        for (const layer of this.selectedLayers.values()) {
          const positiveSlice = slicePair[0];
          const negativeSlice = slicePair[1];
          const conceptSetIds =
              this.sliceService.getSliceByName(positiveSlice)!;
          const config: CallConfig = {
            'concept_set_ids': conceptSetIds,
            'class_to_explain': gradClass,
            'grad_layer': layer,
            'dataset_name': this.appState.currentDataset,
          };
          if (negativeSlice != null) {
            const negativeSliceIds =
                this.sliceService.getSliceByName(negativeSlice)!;
            config['negative_set_ids'] = negativeSliceIds;
          }
          promises.push(
              this.runSingleTCAV(config, positiveSlice, negativeSlice ?? '-'));
        }
      }
    }

    const results = await this.loadLatest('getResults', Promise.all(promises));
    if (results == null) return;

    for (const res of results) {
      if (res == null) continue;
      if (res['config'] == null || res['score'] == null) continue;

      // clang-format off
      let scoreBar: TemplateResult|string = html`<tcav-score-bar
             score=${res['score']}
             meanVal=${res['random_mean']}
             clampVal=${1}>
           </tcav-score-bar>`;
      // clang-format on
      let displayScore = res.score.toFixed(3);

      if (res['p_val'] != null && res['p_val'] > MAX_P_VAL) {
        displayScore = '-';
        scoreBar = HIGH_P_VAL_WARNING;
      }
      if (res['p_val'] == null) {
        scoreBar = NO_T_TESTING_WARNING;
      }


      this.resultsTableData.push({
        'Positive Slice': res['positiveSlice'],
        'Negative Slice': res['negativeSlice'],
        'Run': this.cavCounter,
        'Embedding': res['config']['grad_layer'],
        'Class': res['config']['class_to_explain'],
        'CAV Score': displayScore,
        'Score Bar': scoreBar
      });
    }
    this.cavCounter++;
    this.isLoading = false;
    this.requestUpdate();
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    // Ensure the models can support TCAV and that the TCAV interpreter is
    // loaded.
    const supportsEmbs = doesOutputSpecContain(modelSpecs, 'Embeddings');
    const supportsGrads = doesOutputSpecContain(modelSpecs, 'Gradients');
    const multiclassPreds =
        doesOutputSpecContain(modelSpecs, 'MulticlassPreds');
    if (!supportsGrads || !supportsEmbs || !multiclassPreds) {
      return false;
    }
    for (const modelInfo of Object.values(modelSpecs)) {
      if (modelInfo.interpreters.indexOf('tcav') !== -1) {
        return true;
      }
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'tcav-module': TCAVModule;
  }
}
