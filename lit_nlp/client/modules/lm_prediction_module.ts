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
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {classMap} from 'lit/directives/class-map';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {IndexedInput, ModelInfoMap, Spec, TopKResult} from '../lib/types';
import {findSpecKeys, isLitSubtype} from '../lib/utils';

import {styles} from './lm_prediction_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

/**
 * A LIT module that renders masked predictions for a masked LM.
 */
@customElement('lm-prediction-module')
export class LanguageModelPredictionModule extends LitModule {
  static override title = 'LM Predictions';
  static override duplicateForExampleComparison = true;
  static override duplicateAsRow = true;
  static override numCols = 4;
  static override template = (model = '', selectionServiceIndex = 0) => {
    return html`<lm-prediction-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></lm-prediction-module>`;
  };

  static override get styles() {
    return [sharedStyles, styles];
  }

  // Module options / configuration state
  @observable private clickToMask: boolean = false;

  // Fixed output state (based on unmodified input)
  @observable private selectedInput: IndexedInput|null = null;
  @observable private tokens: string[] = [];
  @observable private originalResults: TopKResult[][] = [];
  // Ephemeral output state (may depend on selectedTokenIndex)
  @observable private selectedTokenIndex: number|null = null;
  @observable private mlmResults: TopKResult[][] = [];

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

  override firstUpdated() {
    const getSelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getSelectedInputData, selectedInput => {
      this.updateSelection(selectedInput);
    });
    this.react(() => this.selectedTokenIndex, tokenIndex => {
      this.updateMLMResults();
    });
    this.react(() => this.clickToMask, clickToMask => {
      this.updateMLMResults();
    });
    // Enable click-to-mask if the model supports it.
    this.reactImmediately(() => this.model, model => {
      if (this.maskToken != null) {
        this.clickToMask = true;
      }
    });
  }

  private async updateSelection(input: IndexedInput|null) {
    this.selectedInput = null;
    this.tokens = [];
    this.originalResults = [];
    this.selectedTokenIndex = null;
    this.mlmResults = [];

    if (input == null) return;

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [input], this.model, dataset, ['Tokens', 'TokenTopKPreds'],
        'Loading tokens');
    const results = await this.loadLatest('modelPreds', promise);
    if (results === null) return;

    const predictions = results[0];
    this.tokens = predictions[this.outputTokensKey];
    this.originalResults = predictions[this.predKey];
    this.mlmResults = this.originalResults;
    this.selectedInput = input;

    // If there's already a mask in the input, jump to that.
    if (this.maskToken != null) {
      const maskIndex = this.tokens.indexOf(this.maskToken);
      if (maskIndex !== -1) {
        // Show fills immediately for the first mask token, if there is one.
        this.selectedTokenIndex = maskIndex;
      }
    }
  }

  private async updateMLMResults() {
    if (this.selectedTokenIndex == null || !this.clickToMask) {
      this.mlmResults = this.originalResults;  // reset
      return;
    }
    if (this.selectedInput == null || this.maskToken == null) {
      return;
    }

    const tokens = [...this.tokens];
    // Short-circuit to avoid an extra inference call if this token
    // is already masked in the original input.
    if (tokens[this.selectedTokenIndex] === this.maskToken) {
      this.mlmResults = this.originalResults;
      return;
    }
    tokens[this.selectedTokenIndex] = this.maskToken;

    // Reset current results.
    this.mlmResults = [];

    const inputData = Object.assign(
        {}, this.selectedInput.data, {[this.inputTokensKey!]: tokens});
    // Use empty id to disable caching on backend.
    const inputs: IndexedInput[] =
        [{'data': inputData, 'id': '', 'meta': {added: true}}];

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        inputs, this.model, dataset, ['TokenTopKPreds']);
    const results = await this.loadLatest('mlmResults', promise);
    if (results === null) return;

    this.mlmResults = results[0][this.predKey];
  }

  override render() {
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

  /* Mode info if masking is available. */
  renderModeInfo() {
    if (this.selectedTokenIndex === null) {
      return null;
    }
    // clang-format off
    return html`
      <span class='mode-info'>
        ${this.clickToMask ? "Showing masked predictions" : "Showing unmasked predictions"}
      </span>
    `;
    // clang-format on
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
          ${this.renderModeInfo()}
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
        } else {
          this.selectedTokenIndex = i;
        }
      };
      const classes = classMap({
        'token': true,
        'token-chip-label': true,
        'selected': i === this.selectedTokenIndex,
        'masked': (i === this.selectedTokenIndex) && (this.maskToken != null) &&
            this.clickToMask,
      });
      return html`<div class=${classes} @click=${handleClick}>${token}</div>`;
    };

    // clang-format off
    return html`
      <div class='input-group'>
        <div class='field-title input-group-title'>
          ${this.tokens.length > 0 ? this.outputTokensKey : null}
        </div>
        <div class="input-words">${this.tokens.map(renderToken)}</div>
      </div>
    `;
    // clang-format on
  }

  renderOutputWords() {
    if (this.selectedTokenIndex === null || this.mlmResults.length === 0) {
      return html`<div id="output-words" class='sidebar'></div>`;
    }

    const renderPred = (pred: TopKResult) => {
      const selectedWordType = this.tokens[this.selectedTokenIndex!];
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
        ${this.mlmResults[this.selectedTokenIndex].map(renderPred)}
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

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
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
