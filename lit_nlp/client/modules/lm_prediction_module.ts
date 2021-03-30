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
import {customElement, html} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {IndexedInput, ModelInfoMap, Spec, TopKResult} from '../lib/types';
import {findSpecKeys, flatten, isLitSubtype} from '../lib/utils';

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
  static numCols = 4;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<lm-prediction-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></lm-prediction-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  @observable private clickToMask: boolean = false;

  @observable private tokens: string[] = [];
  @observable private selectedInput: IndexedInput|null = null;
  // TODO(lit-dev): separate state from "ephemeral" preds in click-to-mask mode
  // from the initial model predictions.
  @observable private maskApplied: boolean = false;
  @observable private lmResults: TopKResult[][] = [];
  @observable private selectedTokenIndex: number|null = null;

  @computed
  private get modelSpec() {
    return this.appState.getModelSpec(this.model);
  }

  @computed
  private get predKey(): string {
    // This list is guaranteed to be non-empty due to checkModule()
    return findSpecKeys(this.modelSpec.output, 'TokenTopKPreds')[0];
  }

  @computed
  private get outputTokensKey(): string {
    // This list is guaranteed to be non-empty due to checkModule()
    return this.modelSpec.output[this.predKey].align as string;
  }

  @computed
  private get inputTokensKey(): string|null {
    // Look for an input field matching the output tokens name.
    if (this.modelSpec.input.hasOwnProperty(this.outputTokensKey) &&
        isLitSubtype(this.modelSpec.input[this.outputTokensKey], 'Tokens')) {
      return this.outputTokensKey;
    }
    return null;
  }

  @computed
  private get maskToken() {
    // Look at metadata for /input/ field matching output tokens name.
    return this.inputTokensKey ?
        this.modelSpec.input[this.inputTokensKey].mask_token :
        undefined;
  }

  firstUpdated() {
    const getSelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getSelectedInputData, selectedInput => {
      this.updateSelection(selectedInput);
    });
  }

  private async updateSelection(selectedInput: IndexedInput|null) {
    this.selectedTokenIndex = null;
    if (selectedInput == null) {
      this.selectedInput = null;
      this.tokens = [];
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
    this.tokens = predictions[this.outputTokensKey];
    this.lmResults = predictions[this.predKey];
    this.selectedInput = selectedInput;

    if (this.maskToken != null) {
      const maskIndex = this.tokens.indexOf(this.maskToken);
      if (maskIndex !== -1) {
        // Show fills immediately for the first mask token, if there is one.
        this.selectedTokenIndex = maskIndex;
      }
    }

    // If there's nothing to show, enable click-to-mask by default.
    if (flatten(this.lmResults).length === 0 && this.maskToken != null) {
      this.clickToMask = true;
    }
  }

  // TODO(lit-dev): unify this codepath with updateSelection()?
  private async updateLmResults(maskIndex: number) {
    if (this.selectedInput == null) return;

    if (this.clickToMask && this.maskToken != null) {
      if (this.inputTokensKey == null) return;
      const tokens = [...this.tokens];
      tokens[maskIndex] = this.maskToken;

      const inputData = Object.assign(
          {}, this.selectedInput.data, {[this.inputTokensKey]: tokens});
      // Use empty id to disable caching on backend.
      const inputs: IndexedInput[] =
          [{'data': inputData, 'id': '', 'meta': {added: true}}];

      const dataset = this.appState.currentDataset;
      const promise = this.apiService.getPreds(
          inputs, this.model, dataset, ['TokenTopKPreds']);
      const lmResults = await this.loadLatest('mlmResults', promise);
      if (lmResults === null) return;

      this.lmResults = lmResults[0][this.predKey];
      this.maskApplied = true;
    }
    this.selectedTokenIndex = maskIndex;
  }

  render() {
    return html`
      <div class='module-container'>
        ${this.renderControls()}
        <div id='main-area' class='module-results-area'>
          ${this.renderInputWords()}
          ${this.renderOutputWords()}
        </div>
      </div>
    `;
  }

  renderControls() {
    // clang-format off
    return html`
      <div class='module-toolbar'>
        ${this.maskToken ? html`
          <lit-checkbox label="Click to mask?"
            ?checked=${this.clickToMask}
            @change=${() => { this.clickToMask = !this.clickToMask; }}
          ></lit-checkbox>
        ` : null}
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
        'token': true,
        'token-chip-label': true,
        'selected': i === this.selectedTokenIndex,
        'masked': (i === this.selectedTokenIndex) && this.maskApplied,
      });
      return html`<div class=${classes} @click=${handleClick}>${token}</div>`;
    };

    // clang-format off
    return html`
      <div class='input-group'>
        <div class='group-title'>
          ${this.tokens.length > 0 ? this.outputTokensKey : null}
        </div>
        <div class="input-words">${this.tokens.map(renderToken)}</div>
      </div>
    `;
    // clang-format on
  }

  renderOutputWords() {
    if (this.selectedTokenIndex === null || this.lmResults === null) {
      return html`<div id="output-words" class='sidebar'></div>`;
    }
    const selectedTokenIndex = this.selectedTokenIndex || 0;

    const renderPred = (pred: TopKResult) => {
      const selectedWordType = this.tokens[selectedTokenIndex];
      const predWordType = pred[0];
      // Convert probability into percent.
      const predProb = (pred[1] * 100).toFixed(1);
      const classes = classMap({
        'output-token': true,
        'token-chip-generated': true,
        'selected': predWordType === selectedWordType,
      });
      // clang-format off
      return html`
        <div class='output-row'>
          <div class=${classes}>${predWordType}</div>
          <div class='token-chip-label output-percent'>${predProb}%</div>
        </div>
      `;
      // clang-format on
    };

    // clang-format off
    return html`
      <div id="output-words" class='sidebar sidebar-shaded'>
        ${this.lmResults[selectedTokenIndex].map(renderPred)}
      </div>
    `;
    // clang-format on
  }

  /**
   * Find available output fields.
   */
  static findTargetFields(outputSpec: Spec): string[] {
    const candidates = findSpecKeys(outputSpec, 'TokenTopKPreds');
    return candidates.filter(k => {
      const align = outputSpec[k].align;
      return align != null && isLitSubtype(outputSpec[align], 'Tokens');
    });
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    for (const modelInfo of Object.values(modelSpecs)) {
      if (LanguageModelPredictionModule.findTargetFields(modelInfo.spec.output)
              .length > 0) {
        return true;
      }
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lm-prediction-module': LanguageModelPredictionModule;
  }
}
