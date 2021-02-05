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
import '../elements/tcav_bar_vis';

import {customElement, html} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {computed, observable} from 'mobx';
import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {CallConfig, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {SliceService} from '../services/services';
import {STARRED_SLICE_NAME} from '../services/slice_service';

import {styles as sharedStyles} from './shared_styles.css';
import {styles} from './tcav_module.css';

const MIN_EXAMPLES_LENGTH = 2;  // minimum examples needed to train the CAV.
const ALL = 'all';
const TCAV_INTERPRETER_NAME = 'tcav';
const CHART_MARGIN = 30;
const CHART_WIDTH = 150;
const CHART_HEIGHT = 150;
const ADDED_WIDTH = 60;

/**
 * The TCAV module.
 */
@customElement('tcav-module')
export class TCAVModule extends LitModule {
  static get styles() {
    return [sharedStyles, styles];
  }
  static title = 'TCAV';
  static numCols = 3;
  static duplicateForModelComparison = true;

  static template = (model = '') => {
    return html`
      <tcav-module model=${model}>
      </tcav-module>`;
  };
  private readonly sliceService = app.getService(SliceService);

  @observable private scores = new Map<string, number>();
  @observable private selectedSlice: string = ALL;
  @observable private selectedLayer: string = '';
  @observable private selectedClass: string = '';
  @observable private isLoading: boolean = false;
  @observable private excludedSlices: string[] = [];

  @computed
  get warningText() {
    return `TCAV scores are not showing for ${
        this.excludedSlices.join(
            ', ')} since it did not pass statistical testing (high p-value).`;
  }

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

  firstUpdated() {
    // Set the first grad key as default in selector.
    if (this.selectedLayer === '' && this.gradKeys.length > 0) {
      this.selectedLayer = this.gradKeys[0];
    }
    // Set first pred class as default in selector.
    if (this.selectedClass === '' && this.predClasses.length > 0) {
      this.selectedClass = this.predClasses[0];
    }
  }

  renderSelector(
      fieldName: string, tooltip: string, handleChange: (e: Event) => void,
      selected: string, items: string[], includeAll: boolean = false) {
    // clang-format off
    return html `
        <div class='dropdown-holder'>
          <div class='dropdown-label' title=${tooltip}>${fieldName}</div>
          <select class="tcav-dropdown dropdown" @change=${handleChange}>
            ${includeAll ? html`<option value="${ALL}">${ALL}</option>`: ''}
            ${items.map(val => {
                return html`
                    <option value="${val}"
                    ?selected=${val === selected}>${val}</option>
                    `;
            })}
          </select>
        </div>
        `;
    // clang-format on
  }

  renderSelectors() {
    const handleSliceChange = (e: Event) => {
      const selected = e.target as HTMLInputElement;
      this.selectedSlice = selected.value;
    };
    const handleLayerChange = (e: Event) => {
      const selected = e.target as HTMLInputElement;
      this.selectedLayer = selected.value;
    };
    const handleClassChange = (e: Event) => {
      const selected = e.target as HTMLInputElement;
      this.selectedClass = selected.value;
    };

    // clang-format off
    return html`
        ${this.renderSelector('Slice', 'Slice to create CAV from', handleSliceChange, this.selectedSlice,
                              this.TCAVSliceNames, true)}
        ${this.renderSelector('Embedding', 'Embedding to create CAV for', handleLayerChange,
                              this.selectedLayer, this.gradKeys)}
        ${this.renderSelector('Explain Class', 'Class to explain', handleClassChange,
                              this.selectedClass, this.predClasses)}
        `;
    // clang-format on
  }

  renderSpinner() {
    return html`
      <div class="spinner-container">
        <lit-spinner size=${24} color="var(--app-secondary-color)">
        </lit-spinner>
      </div>
    `;
  }

  render() {
    const shouldDisable = () => {
      const slices = (this.selectedSlice === ALL) ? this.TCAVSliceNames :
                                                    [this.selectedSlice];
      for (const slice of slices) {
        const examples = this.sliceService.getSliceByName(slice);
        if (examples == null) return true;
        if (examples.length >= MIN_EXAMPLES_LENGTH) {
          return false;  // only enable if slice has minimum number of examples
        }
      }
      return true;
    };

    // The width of the SVG increase by 60px for each additional entry after
    // the first bar, so their labels don't overlap.
    // clang-format off
    // TODO(lit-dev): Switch the current barchart viz to a table-based viz.
    return html`
      <div id="outer-container">
        <div class="controls-holder">
          ${this.renderSelectors()}
          <button id='submit' title=${shouldDisable() ?
            `select a slice with ${MIN_EXAMPLES_LENGTH} or more examples`: ''}
            @click=${() => this.runTCAV()} ?disabled=${
             shouldDisable()}>Run TCAV</button>
        </div>
        <div id='vis-container' class=${classMap({'loading': this.isLoading})}>
          ${this.isLoading ? this.renderSpinner(): ''}
          <tcav-bar-vis
            .scores=${this.scores}
            .width=${CHART_WIDTH +
                Math.max(0, this.scores.size - 1) * ADDED_WIDTH}
            .height=${CHART_HEIGHT}
            .margin=${CHART_MARGIN}
          ></tcav-bar-vis>
        </div>
        <div id='warning-text'>
          ${this.excludedSlices.length === 0 ? '': this.warningText}
        </div>
      </div>
    `;
    // clang-format on
  }

  private async runTCAVBySlice(selectedIds: string[], name: string) {
    if (selectedIds.length < MIN_EXAMPLES_LENGTH) {
      return;
    }

    const config: CallConfig = {
      'concept_set_ids': selectedIds,
      'class_to_explain': this.selectedClass,
      'grad_layer': this.selectedLayer,
    };

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
    // Only shows examples with a p-value less than 0.05.
    // TODO(lit-dev): Add display text when the concepts have high p-values.
    // TODO(lit-dev): Show local TCAV scores in the scalar chart.
    if (result[0]['p_val'] < 0.05) {
      const score = result[0]['result']['score'];
      // TODO(lit-dev): Wrap the axis text, or switch to an alternative.
      // layout (e.g. similar to the table in Classification Results).
      const axisLabel =
          `${name} (${this.selectedLayer}, ${this.selectedClass})`;
      return {'axisLabel': axisLabel, 'score': score};
    }
    this.excludedSlices.push(name);
    return {};
  }

  private async runTCAV() {
    this.isLoading = true;
    this.excludedSlices = [];
    const scores = new Map<string, number>(this.scores);

    // TODO(lit-dev): Add option to run TCAV on selected examples.
    // TODO(lit-dev): Add option to run TCAV on categorical features.
    // Run TCAV for all slices if 'all' is selected.
    const slicesToRun =
        this.selectedSlice === ALL ? this.TCAVSliceNames : [this.selectedSlice];
    const sliceInfo = slicesToRun.map((slice) => {
      const selectedIds = this.sliceService.getSliceByName(slice)!;
      return {'name': slice, 'ids': selectedIds};
    });

    const promises = sliceInfo.map(
        slice => this.runTCAVBySlice(slice['ids'], slice['name']));
    const results = await this.loadLatest('getResults', Promise.all(promises));
    if (results == null) return;

    for (const res of results) {
      if (res == null) continue;
      if (res['axisLabel'] == null || res['score'] == null) continue;
      scores.set(res['axisLabel'], res['score']);
    }
    this.isLoading = false;
    this.scores = scores;
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
