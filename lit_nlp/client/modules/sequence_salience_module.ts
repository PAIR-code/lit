/**
 * @fileoverview Visualization for seq2seq salience maps.
 */

// tslint:disable:no-new-decorators
import {customElement, html} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {IndexedInput, ModelInfoMap, Spec} from '../lib/types';

import {UnsignedSalienceCmap} from './salience_map_module';
import {styles} from './sequence_salience_module.css';

interface SequenceSalienceMap {
  sourceTokens: string[];
  targetTokens: string[];
  /** [target_tokens.length, source_tokens.length + target_tokens.length] */
  salience: number[][];
}

interface TokenFocusState {
  fieldName: string;
  idx: number; /* output token index */
  sticky: boolean;
}

/** LIT module for model output. */
@customElement('sequence-salience-module')
export class SequenceSalienceModule extends LitModule {
  static title = 'Sequence Salience';
  static duplicateForExampleComparison = true;
  static duplicateForModelComparison = true;
  static numCols = 4;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`
      <sequence-salience-module model=${model}
       selectionServiceIndex=${selectionServiceIndex}>
      </sequence-salience-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  // Current data
  @observable private currentData?: IndexedInput;
  @observable
  private currentPreds: {[fieldName: string]: SequenceSalienceMap} = {};
  @observable private focusState?: TokenFocusState = undefined;

  // Options
  @observable private cmapGamma: number = 0.5;

  @computed
  get cmap() {
    return new UnsignedSalienceCmap(/* gamma */ this.cmapGamma);
  }

  override firstUpdated() {
    const getPrimarySelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getPrimarySelectedInputData, data => {
      this.updateToSelection(data);
    });
  }

  private async updateToSelection(input: IndexedInput|null) {
    if (input == null) {
      this.currentData = undefined;
      this.currentPreds = {};
      return;
    }
    // Before waiting for the backend call, update data and clear annotations.
    this.currentData = input;
    this.currentPreds = {};  // empty preds will render as (no data)

    const name = 'sequence_salience';
    const promise = this.apiService.getInterpretations(
        [input], this.model, this.appState.currentDataset, name, {},
        `Running ${name}`);
    const results = await this.loadLatest('salience', promise);
    if (results === null) return;

    const preds = results[0];
    const processedPreds: {[fieldName: string]: SequenceSalienceMap} = {};
    for (const fieldName of Object.keys(preds)) {
      processedPreds[`${name}:${fieldName}`] = {
        sourceTokens: preds[fieldName]['tokens_in'],
        targetTokens: preds[fieldName]['tokens_out'],
        salience: preds[fieldName]['salience'],
      };
    }
    // Update data again, in case selection changed rapidly.
    this.currentData = input;
    this.currentPreds = processedPreds;
  }

  renderField(fieldName: string) {
    const preds = this.currentPreds[fieldName];
    const sourceFieldName = 'Source';
    const targetFieldName = 'Target';

    // Output token to show salience for.
    const focusIdx = this.focusState?.fieldName === fieldName ?
        this.focusState?.idx ?? -1 :
        -1;

    const cmap = this.cmap;
    // length is preds.sourceTokens.length + preds.targetTokens.length
    const salience: number[] = focusIdx > -1 ? preds.salience[focusIdx] :
                                               [...preds.salience[0]].fill(0);

    const onMouseoverTarget = (idx: number) => {
      if (this.focusState?.sticky) return;
      this.focusState = {fieldName, idx, sticky: false};
    };

    const onMouseoutTarget = () => {
      if (this.focusState?.sticky) return;
      this.focusState = undefined;
    };

    const onClickTarget = (idx: number, e: Event) => {
      e.stopPropagation(); /* don't trigger focus clear from background */
      if (this.focusState?.fieldName === fieldName &&
          this.focusState?.idx === idx && this.focusState?.sticky) {
        // Only clear if match and sticky, since usually this will already
        // be the non-sticky focus due to mouseover.
        this.focusState = undefined; /* clear focus */
      } else {
        this.focusState = {fieldName, idx, sticky: true};
      }
    };

    // clang-format off
    return html`
      <div class='field-group'>
        <div class="field-title">${fieldName}</div>
        <table class='field-table'>
          <tr>
            <th><div class='subfield-title'>${sourceFieldName}</div></th>
            <td><div class='token-holder'>
                ${preds.sourceTokens.map((token, i) => {
                  const val = salience[i];
                  const tokenStyle = styleMap({
                    'color': cmap.textCmap(val),
                    'background-color': cmap.bgCmap(val)
                  });
                  const classes = classMap({
                    'token': true,
                    'salient-token': true,
                    'source-token': true,
                  });
                  return html`<div class=${classes} style=${tokenStyle}>
                                ${token}
                              </div>`;
                })}
            </div></td>
          </tr>
          <tr>
            <th><div class='subfield-title'>${targetFieldName}</div></th>
             <td><div class='token-holder'>
                <div class='salient-token token-spacer'>&gt;&gt;</div>
                ${preds.targetTokens.map((token, i) => {
                  const val = salience[i + preds.sourceTokens.length];
                  const tokenStyle = styleMap({
                    'color': cmap.textCmap(val),
                    'background-color': cmap.bgCmap(val)
                  });
                  const tokenClasses = classMap({
                    'token': true,
                    'salient-token': true,
                    'token-focused': i === focusIdx,
                    'target-token': true,
                  });
                  return html`<div class=${tokenClasses} style=${tokenStyle}
                               @mouseover=${() => { onMouseoverTarget(i); }}
                               @mouseout=${onMouseoutTarget}
                               @click=${(e: Event) => { onClickTarget(i, e); }}>
                                ${token}
                              </div>`;
                })}
            </div></td>
          </tr>
        </table>
      </div>
    `;
    // clang-format on
  }

  renderContent() {
    if (!this.currentData) return null;

    return Object.keys(this.currentPreds).map(k => this.renderField(k));
  }

  getStatus() {
    const fs = this.focusState;
    if (fs === undefined) return null;
    if (fs.sticky) {
      const token = this.currentPreds[fs.fieldName].targetTokens[fs.idx];
      return `Pinned to token ${fs.idx} "${token}" of ${fs.fieldName}.`;
    }
    return null;
  }

  renderFooterControls() {
    // TODO(b/198684817): move this colormap impl to a shared element?

    const onChangeGamma = (e: Event) => {
      this.cmapGamma = +((e.target as HTMLInputElement).value);
    };

    const cmap = this.cmap;
    const blockValues = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    // clang-format off
    return html`
      <div class='controls'>
        <div class="controls-group">
          <label>Colormap:</label>
          ${blockValues.map(val => {
            const blockStyle = styleMap({
              'color': cmap.textCmap(val),
              'background-color': cmap.bgCmap(val)
            });
            return html`
              <div class='cmap-block' style=${blockStyle}>
                ${val.toFixed(1)}
              </div>`;
          })}
        </div>
        <div class="controls-group">
          <label for="gamma-slider">Gamma:</label>
          <input type="range" min=0.25 max=4 step=0.25
            .value=${this.cmapGamma.toString()} class="slider" id="gamma-slider"
             @input=${onChangeGamma}>
          <div class="gamma-value">${this.cmapGamma.toFixed(2)}</div>
        </div>
      </div>`;
    // clang-format on
  }

  override render() {
    const clearStickyFocus = () => {
      if (this.focusState?.sticky) {
        this.focusState = undefined; /* clear focus */
      }
    };

    // clang-format off
    return html`
      <div class="module-container">
        <div class="module-content" @click=${clearStickyFocus}>
          ${this.renderContent()}
        </div>
        <div class="module-footer">
          <p class="module-status">${this.getStatus()}</p>
          ${this.renderFooterControls()}
        </div>
      </div>
    `;
    // clang-format on
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    // TODO(b/177963928): determine visibility based on component list
    // and model compatibility?
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'sequence-salience-module': SequenceSalienceModule;
  }
}
