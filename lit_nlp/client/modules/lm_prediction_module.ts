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

import '../elements/checkbox';
// tslint:disable:no-new-decorators
import {customElement, html, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {IndexedInput, ModelsMap, Spec, TopKResult} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, flatten} from '../lib/utils';

import {styles} from './lm_prediction_module.css';
import {styles as sharedStyles} from './shared_styles.css';

/**
 * A LIT module that renders masked predictions for a masked LM.
 */
@customElement('lm-prediction-module')
export class LanguageModelPredictionModule extends LitModule {
  static title = 'LM Predictions';
  static duplicateForExampleComparison = true;
  static duplicateAsRow = true;
  static numCols = 6;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<lm-prediction-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></lm-prediction-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  // TODO(lit-dev): get this from the model spec?
  @property({type: String}) maskToken: string = '[MASK]';

  @observable private clickToMask: boolean = false;

  @observable private tokens: string[] = [];
  @observable private selectedInput: IndexedInput|null = null;
  // TODO(lit-dev): separate state from "ephemeral" preds in click-to-mask mode
  // from the initial model predictions.
  @observable private maskApplied: boolean = false;
  @observable private lmResults: TopKResult[][] = [];
  @observable private selectedTokenIndex: number|null = null;

  @computed
  private get predKey(): string {
    const spec = this.appState.getModelSpec(this.model);
    // This list is guaranteed to be non-empty due to checkModule()
    return findSpecKeys(spec.output, 'TokenTopKPreds')[0];
  }

  @computed
  private get outputTokenKey(): string {
    const spec = this.appState.getModelSpec(this.model);
    // This list is guaranteed to be non-empty due to checkModule()
    return spec.output[this.predKey].align as string;
  }

  @computed
  private get inputTextKey(): string {
    const spec = this.appState.getModelSpec(this.model);
    // TODO(lit-dev): ensure this is set in order to enable MLM mode.
    return spec.output[this.outputTokenKey].parent!;
  }

  firstUpdated() {
    const getSelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getSelectedInputData, selectedInput => {
      this.updateSelection(selectedInput);
    });
  }

  private async updateSelection(selectedInput: IndexedInput|null) {
    if (selectedInput == null) {
      this.selectedInput = null;
      this.tokens = [];
      this.selectedTokenIndex = null;
      this.lmResults = [];
      return;
    }

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [selectedInput], this.model, dataset, ['Tokens', 'TokenTopKPreds'],
        'Loading tokens');
    const results = await this.loadLatest('modelPreds', promise);
    if (results === null) return;

    const predictions = results[0];
    this.tokens = predictions[this.outputTokenKey];
    this.lmResults = predictions[this.predKey];
    this.selectedInput = selectedInput;

    const maskIndex = this.tokens.indexOf(this.maskToken);
    if (maskIndex !== -1) {
      // Show fills immediately for the first mask token, if there is one.
      this.selectedTokenIndex = maskIndex;
    }

    // If there's nothing to show, enable click-to-mask by default.
    // TODO(lit-dev): infer this from something in the spec instead.
    if (flatten(this.lmResults).length === 0) {
      this.clickToMask = true;
    }
  }

  // TODO(lit-dev): unify this codepath with updateSelection()?
  private async updateLmResults(index: number) {
    if (this.selectedInput == null) return;

    if (this.clickToMask) {
      const tokens = [...this.tokens];
      tokens[index] = this.maskToken;
      // TODO(lit-dev): detokenize properly, or feed tokens directly to model?
      const input = tokens.join(' ');

      // Use empty id to disable caching on backend.
      const inputData = Object.assign(
          {}, this.selectedInput.data, {[this.inputTextKey]: input});
      const inputs: IndexedInput[] =
          [{'data': inputData, 'id': '', 'meta': {}}];

      const dataset = this.appState.currentDataset;
      const promise = this.apiService.getPreds(
          inputs, this.model, dataset, ['TokenTopKPreds']);
      const lmResults = await this.loadLatest('mlmResults', promise);
      if (lmResults === null) return;

      this.lmResults = lmResults[0][this.predKey];
      this.maskApplied = true;
    }
    this.selectedTokenIndex = index;
  }

  updated() {
    if (this.selectedTokenIndex == null) return;

    // Set the correct offset for displaying the predicted tokens.
    const inputTokenDivs = this.shadowRoot!.querySelectorAll('.token');
    const maskedInputTokenDiv =
        inputTokenDivs[this.selectedTokenIndex] as HTMLElement;
    const offsetX = maskedInputTokenDiv.offsetLeft;
    const outputTokenDiv =
        this.shadowRoot!.getElementById('output-words') as HTMLElement;
    outputTokenDiv.style.marginLeft = `${offsetX - 8}px`;
  }

  render() {
    return html`
      ${this.renderControls()}
      ${this.renderInputWords()}
      ${this.renderOutputWords()}
    `;
  }

  renderControls() {
    // TODO: check if MLM is applicable.
    // clang-format off
    return html`
      <div id='controls'>
        <lit-checkbox label="Click to mask?"
          ?checked=${this.clickToMask}
          @change=${() => { this.clickToMask = !this.clickToMask; }}
        ></lit-checkbox>
      </div>
    `;
    // clang-format on
  }

  renderInputWords() {
    const renderToken = (token: string, i: number) => {
      const handleClick = () => {
        if (i === this.selectedTokenIndex) {
          // Clear if the same position is clicked again.
          this.selectedTokenIndex = null;
          this.maskApplied = false;
        } else {
          this.updateLmResults(i);
        }
      };
      const classes = classMap({
        token: true,
        selected: i === this.selectedTokenIndex,
        masked: (i === this.selectedTokenIndex) && this.maskApplied,
      });
      return html`<div class=${classes} @click=${handleClick}>${token}</div>`;
    };

    // clang-format on
    return html`
      <div id="input-words">
        ${this.tokens.map(renderToken)}
      </div>
    `;
    // clang-format off
  }

  renderOutputWords() {
    if (this.selectedTokenIndex === null || this.lmResults === null) {
      return html``;
    }
    const selectedTokenIndex = this.selectedTokenIndex || 0;

    const renderPred = (pred: TopKResult) => {
      const selectedWordType = this.tokens[selectedTokenIndex];
      const predWordType = pred[0];
      // Convert probability into percent.
      const predProb = (pred[1] * 100).toFixed(1);
      const classes = classMap({
        output: true,
        same: predWordType === selectedWordType,
      });
      return html`<div class=${classes}>${predWordType} ${predProb}%</div>`;
    };

    // clang-format off
    return html`
      <div id="output-words">
        ${this.lmResults[selectedTokenIndex].map(renderPred)}
      </div>
    `;
    // clang-format on
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    // TODO(lit-dev): check for tokens field here, else may crash if not
    // present.
    return doesOutputSpecContain(modelSpecs, 'TokenTopKPreds');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lm-prediction-module': LanguageModelPredictionModule;
  }
}
