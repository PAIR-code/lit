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
import {TextSegment, Tokens, TokenTopKPreds} from '../lib/lit_types';
import {IndexedInput, ModelInfoMap, Spec, TopKResult} from '../lib/types';
import {findMatchingIndices, findSpecKeys, replaceNth} from '../lib/utils';

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
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <lm-prediction-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </lm-prediction-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  // Module options / configuration state
  @observable private clickToMask: boolean = false;

  // Fixed output state (based on unmodified input)
  @observable private selectedInput: IndexedInput|null = null;
  @observable private tokens: string[] = [];
  @observable private maskedInput: IndexedInput|null = null;
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
    return findSpecKeys(this.modelSpec.output, TokenTopKPreds)[0];
  }

  @computed
  private get outputTokensKey(): string {
    // This list is guaranteed to be non-empty due to checkModule()
    return (this.modelSpec.output[this.predKey] as TokenTopKPreds).align as
        string;
  }

  @computed
  private get outputTokensPrefix(): string {
    // This list is guaranteed to be non-empty due to checkModule()
    return (this.modelSpec.output[this.outputTokensKey] as Tokens)
               .token_prefix as string;
  }

  @computed
  private get inputTokensKey(): string|null {
    // Look for an input field matching the output tokens name.
    if (this.modelSpec.input.hasOwnProperty(this.outputTokensKey) &&
        this.modelSpec.input[this.outputTokensKey] instanceof Tokens) {
      return this.outputTokensKey;
    }
    return null;
  }

  @computed
  private get maskToken() {
    // Look at metadata for /input/ field matching output tokens name.
    return this.inputTokensKey ?
        (this.modelSpec.input[this.inputTokensKey] as Tokens).mask_token :
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
    this.maskedInput =  null;
    this.originalResults = [];
    this.selectedTokenIndex = null;
    this.mlmResults = [];

    if (input == null) return;

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [input], this.model, dataset, [Tokens, TokenTopKPreds], [],
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

  private createChildDatapoint(orig: IndexedInput, tokens: string[]) {
    const inputData = Object.assign(
        {}, orig.data, {[this.inputTokensKey!]: tokens});
    return {
      data: inputData,
      id: '',
      meta: {
        added: true,
        source: 'masked',
        parentId: orig.id
      }
    };
  }

  private async updateMLMResults() {
    if (this.selectedTokenIndex == null || !this.clickToMask) {
      this.mlmResults = this.originalResults;  // reset
      this.maskedInput =  null;
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

    // Create masked datapoint.
    tokens[this.selectedTokenIndex] = this.maskToken;
    this.maskedInput = this.createChildDatapoint(this.selectedInput, tokens);
    this.updateTextSegment(
        this.maskedInput, this.maskToken, this.selectedTokenIndex);

    // Reset current results.
    this.mlmResults = [];

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [this.maskedInput], this.model, dataset, [TokenTopKPreds]);
    const results = await this.loadLatest('mlmResults', promise);
    if (results === null) return;

    this.mlmResults = results[0][this.predKey];
  }

  /** Update the datapoint's TextSegment based on a new token value. */
  private updateTextSegment(datapoint: IndexedInput, token: string,
                            tokenIndex: number) {
    // This logic ensure that if the token to replace occurs multiple times
    // in the text segment, that the correct instance of the token is replaced.
    const textField = findSpecKeys(this.modelSpec.input, TextSegment)[0];
    let oldToken = this.tokens[tokenIndex];
    const tokensIndicesMatchingToken = findMatchingIndices(
        this.tokens, oldToken);
    const replacementIndex = tokensIndicesMatchingToken.indexOf(tokenIndex);
    if (this.outputTokensPrefix != null &&
        oldToken.startsWith(this.outputTokensPrefix)) {
      oldToken = oldToken.slice(this.outputTokensPrefix.length);
    }
    const newText = replaceNth(datapoint.data[textField], oldToken, token,
                               replacementIndex + 1);
    datapoint.data[textField] = newText;
  }

  override renderImpl() {
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
        ${this.clickToMask ? "Masked predictions" : "Unmasked predictions"}
      </span>
    `;
    // clang-format on
  }

  renderAddInputButton() {
    const onClickAdd = async () => {
      const data: IndexedInput[] = await this.appState.annotateNewData(
          [this.maskedInput!]);
      this.appState.commitNewDatapoints(data);
    };
    return html`
      <button class='hairline-button'
        @click=${onClickAdd} ?disabled="${this.maskedInput == null}">
        Add masked datapoint
      </button>`;
  }

  renderControls() {
    // clang-format off
    return html`
      <div class='module-toolbar'>
        ${this.maskToken ? html`
          <lit-checkbox label="Mask selected token"
            ?checked=${this.clickToMask}
            @change=${() => { this.clickToMask = !this.clickToMask; }}
          ></lit-checkbox>` : null}
        ${this.renderAddInputButton()}
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
        <div class='field-title input-group-title' title='Field from model'>
          ${this.tokens.length > 0 ? this.outputTokensKey : null}
        </div>
        <div class="input-words">${this.tokens.map(renderToken)}</div>
      </div>
    `;
    // clang-format on
  }

  renderOutputWords() {

    const renderPred = (pred: TopKResult) => {
      const selectedWordType = this.tokens[this.selectedTokenIndex!];
      const predWordType = pred[0];
      // Convert probability into percent.
      const predProb = (pred[1] * 100).toFixed(1);
      const matchesInput = predWordType === selectedWordType;
      const classes = classMap({
        'output-token': true,
        'token-chip-generated': true,
        'selected': matchesInput,
      });
      const addPoint = async () => {
        // Create new datapoint with selected replacement token and add to
        // dataset.
        const tokens = [...this.tokens];
        const newInput = this.createChildDatapoint(this.selectedInput!, tokens);
        newInput.data[this.inputTokensKey!][this.selectedTokenIndex!] =
            predWordType;
        this.updateTextSegment(
            newInput, predWordType, this.selectedTokenIndex!);

        const data: IndexedInput[] = await this.appState.annotateNewData(
          [newInput]);
        this.appState.commitNewDatapoints(data);
      };
      // clang-format off
      return html`
        <div class='output-row'>
          <div class='flex-holder'>
            <mwc-icon class="icon-button outlined add-icon-button"
                      @click=${addPoint} ?disabled="${matchesInput}"
                      title="Add to dataset">
              add_box
            </mwc-icon>
            <div class=${classes}>${predWordType}</div>
          </div>
          <div class='token-chip-label output-percent'>${predProb}%</div>
        </div>
      `;
      // clang-format on
    };

    const renderSidebarContents = () => {
      if (this.selectedTokenIndex === null || this.mlmResults.length === 0) {
        return html`
            <span class='mode-info'>
              Click a token to see predictions
            </span>`;
      } else {
        // TODO(b/210998285): Add surrounding token context above predictions.
        return html`
            ${this.maskToken ? html`${this.renderModeInfo()}` : null}
            ${this.mlmResults[this.selectedTokenIndex].map(renderPred)}`;
      }
    };

    // clang-format off
    return html`
      <div id="output-words" class='sidebar sidebar-shaded'>
        ${renderSidebarContents()}
      </div>
    `;
    // clang-format on
  }

  /**
   * Find available output fields.
   */
  static findTargetFields(outputSpec: Spec): string[] {
    const candidates = findSpecKeys(outputSpec, TokenTopKPreds);
    return candidates.filter(k => {
      const align = (outputSpec[k] as TokenTopKPreds).align;
      return align != null && (outputSpec[align] instanceof Tokens);
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
