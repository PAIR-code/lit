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

/**
 * Module within LIT the model's attention for a single input
 */

// tslint:disable:no-new-decorators
import {css, html, svg} from 'lit';
import {customElement} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {getBrandColor} from '../lib/colors';
import {AttentionHeads as AttentionHeadsLitType, Tokens as TokensLitType} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {IndexedInput, ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, getTextWidth, getTokOffsets, sumArray} from '../lib/utils';
import {FocusService} from '../services/services';

type Tokens = string[];
// <float>[num_heads, num_tokens, num_tokens]
type AttentionHeads = number[][][];

/**
 * A LIT module that renders the model's attention for a single input.
 */
@customElement('attention-module')
export class AttentionModule extends LitModule {
  static override title = 'Attention';
  static override referenceURL =
      'https://github.com/PAIR-code/lit/wiki/components.md#attention';
  static override numCols = 3;
  static override collapseByDefault = true;
  static override duplicateForExampleComparison = true;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => {
        return html`
      <attention-module model=${model} .shouldReact=${shouldReact}
        selectionServiceIndex=${selectionServiceIndex}>
      </attention-module>`;
      };

  static override get styles() {
    const styles = css`
        .head-selector-chip {
          margin: 0px 1px;
          width: 1rem;
          text-align: center;
        }

        .head-selector-chip.selected {
          color: var(--lit-cyea-600);
          border-color: var(--lit-cyea-600);
        }

        .head-selector-chip:hover {
          background: var(--lit-cyea-100);
          cursor: pointer;
        }

        .padded-container {
          padding: 4px 8px;
        }
    `;
    return [sharedStyles, styles];
  }

  private readonly focusService = app.getService(FocusService);
  private clearFocusTimer: number|undefined;

  @observable private selectedLayer?: string;
  @observable private selectedHeadIndex: number = 0;
  @observable private preds?: {[key: string]: Tokens|AttentionHeads};

  override firstUpdated() {
    const getAttnInputs = () =>
        [this.selectionService.primarySelectedInputData, this.selectedLayer];
    this.reactImmediately(getAttnInputs, (selectedInput, selectedLayer) => {
      this.updateSelection(
          this.selectionService.primarySelectedInputData, this.selectedLayer!);
    });
  }

  private async updateSelection(
      selectedInput: IndexedInput|null, layer: string) {
    this.preds = undefined;  // clear previous results

    if (selectedInput === null) return;
    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [selectedInput], this.model, dataset, [TokensLitType], [layer],
        'Fetching attention');
    const res = await this.loadLatest('attentionAndTokens', promise);
    if (res === null) return;
    this.preds = res[0];
    // Make sure head selection is valid.
    const numHeadsPerLayer =
        this.preds[layer] != null ? this.preds[layer].length : 0;
    if (this.selectedHeadIndex >= numHeadsPerLayer) {
      this.selectedHeadIndex = 0;
    }
  }

  override renderImpl() {
    // Scrolling inside this module is done inside the module-results-area div.
    // Giving this div the class defined by SCROLL_SYNC_CSS_CLASS allows
    // scrolling to be sync'd instances of this module when doing comparisons
    // between models and/or duplicated datapoints. See lit_module.ts for more
    // details.
    // clang-format off
    return html`
      <div class='module-container'>
        <div class='module-toolbar'>
          ${this.renderLayerSelector()}
          ${this.renderHeadSelector()}
        </div>
        <div class='module-results-area padded-container ${SCROLL_SYNC_CSS_CLASS}'>
          ${this.preds != null ? this.renderAttnHead(): null}
        </div>
      </div>
    `;
    // clang-format on
  }

  private renderAttnHead() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const fieldSpec =
        outputSpec[this.selectedLayer!] as AttentionHeadsLitType;

    // Tokens involved in the attention.
    const inToks = (this.preds!)[fieldSpec.align_in!] as Tokens;
    const outToks = (this.preds!)[fieldSpec.align_out!] as Tokens;

    const fontFamily = '\'Share Tech Mono\', monospace';
    const fontSize = 12;
    const defaultCharWidth = 6.5;
    const font = `${fontSize}px ${fontFamily}`;
    const inTokWidths = inToks.map(tok => getTextWidth(tok, font, defaultCharWidth));
    const outTokWidths = outToks.map(tok => getTextWidth(tok, font, defaultCharWidth));
    const spaceWidth = getTextWidth(' ', font, defaultCharWidth);

    // Height of the attention visualization part.
    const visHeight = 100;

    // Vertical pad between attention vis and words.
    const pad = 10;

    // Calculate the full width and height.
    const inTokWidth = sumArray(inTokWidths) + (inToks.length - 1) * spaceWidth;
    const outTokWidth = sumArray(outTokWidths) + (outToks.length - 1) * spaceWidth;
    const width = Math.max(inTokWidth, outTokWidth);
    const height = visHeight + fontSize * 2 + pad * 4;
    const inTokOffsets = getTokOffsets(inTokWidths, spaceWidth);
    const outTokOffsets = getTokOffsets(outTokWidths, spaceWidth);

    // If focus is one any of the tokens in the attention viz, then only show
    // attention info for those tokens.
    const focusData = this.focusService.focusData;
    let inTokenIdxFocus = null;
    let outTokenIdxFocus =  null;
    const primaryDatapointFocused = focusData != null &&
        focusData.datapointId ===
          this.selectionService.primarySelectedInputData!.id;
    if (primaryDatapointFocused &&
        (focusData!.fieldName === fieldSpec.align_out ||
         focusData!.fieldName === fieldSpec.align_in)) {
      inTokenIdxFocus = focusData!.fieldName === fieldSpec.align_in
          ? focusData!.subField: -1;
      outTokenIdxFocus = focusData!.fieldName === fieldSpec.align_out
          ? focusData!.subField: -1;
    }
    const clearFocus = () => {
      if (this.clearFocusTimer != null) {
        clearTimeout(this.clearFocusTimer);
      }
      this.clearFocusTimer = setTimeout(() => {
        this.focusService.clearFocus();
      }, 500) as unknown as number;
    };

    const toksRender = (tok: string, i: number, isInputToken: boolean) => {
      const alignVal =
          isInputToken ? fieldSpec.align_in! : fieldSpec.align_out!;
      const mouseOver = () => {
        clearTimeout(this.clearFocusTimer);
        this.focusService.setFocusedField(
            this.selectionService.primarySelectedInputData!.id,
            'input',
            alignVal,
            i);
      };
      const mouseOut = () => {
        clearFocus();
      };
      const x = isInputToken ? inTokOffsets[i] : outTokOffsets[i];
      const text = svg`${tok}`;
      let opacity = 1;
      if (primaryDatapointFocused &&
          focusData!.fieldName === alignVal &&
          focusData!.subField !== i) {
        opacity = 0.2;
      }

      const y = isInputToken ? visHeight + 4 * pad : pad * 2;
      return svg`<text y=${y} x=${x} opacity=${opacity}
          @mouseover=${mouseOver} @mouseout=${mouseOut}> ${text}</text>`;
    };

    // clang-format off
    return svg`
    <svg width=${width} height=${height}
      font-family="${fontFamily}"
      font-size="${fontSize}px">
      ${outToks.map((tok, i) => toksRender(tok, i, false))}
      ${this.renderAttnLines(visHeight, spaceWidth, 2.5 * pad, inTokWidths,
                             outTokWidths, inTokenIdxFocus, outTokenIdxFocus)}
      ${inToks.map((tok, i) => toksRender(tok, i, true))}
    </svg>
    `;
    // clang-format on
  }

  /**
   * Render the actual lines between tokens to show the attention values.
   */
  private renderAttnLines(
      visHeight: number, spaceWidth: number, pad: number, inTokWidths: number[],
      outTokWidths: number[], inTokenIdxFocus: number|null,
      outTokenIdxFocus: number|null) {
    const inTokOffsets = getTokOffsets(inTokWidths, spaceWidth);
    const outTokOffsets = getTokOffsets(outTokWidths, spaceWidth);
    const y1 = pad;
    const y2 = pad + visHeight;

    const xIn = (i: number) => inTokOffsets[i] + (inTokWidths[i] / 2);
    const xOut = (i: number) => outTokOffsets[i] + (outTokWidths[i] / 2);

    const heads = this.preds![this.selectedLayer!] as AttentionHeads;

    // clang-format off
    return heads[this.selectedHeadIndex].map(
        (attnVals: number[], i: number) => {
          return svg`
            ${attnVals.map((attnVal: number, j: number) => {
              // If token focus index is not null and not equal to an endpoint
              // of an attention line, then do not render it.
              if (inTokenIdxFocus != null && outTokenIdxFocus != null &&
                  (i !== inTokenIdxFocus && j !== outTokenIdxFocus)) {
                return null;
              } else {
                return svg`
                  <line
                    x1=${xOut(i)}
                    y1=${y1}
                    x2=${xIn(j)}
                    y2=${y2}
                    stroke="${getBrandColor('cyea', '600').color}"
                    stroke-opacity="${attnVal}"
                    stroke-width=2>
                  </line>`;
              }
          })}`;
        });
    // clang-format on
  }

  /**
   * Render the dropdown with the layer names.
   */
  private renderLayerSelector() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const attnKeys = findSpecKeys(outputSpec, AttentionHeadsLitType);
    if (this.selectedLayer === undefined) {
      this.selectedLayer = attnKeys[0];
    }
    if (this.preds == null) {
      return;
    }
    const onchange = (e: Event) => {
      this.selectedLayer = (e.target as HTMLSelectElement).value;
    };
    // clang-format off
    return html`
      <select class="dropdown" @change=${onchange}>
        ${attnKeys.map(key =>
          html`<option value=${key} ?selected=${key === this.selectedLayer}>
                 ${key}
               </option>`)}
      </select>
    `;
    // clang-format on
  }

  /**
   * Render the dropdown for the attention head index.
   */
  private renderHeadSelector() {
    const renderChip = (i: number) => {
      const handleClick = () => {
        this.selectedHeadIndex = i;
      };
      const classes = classMap({
        'head-selector-chip': true,
        'token-chip-function': true,
        'selected': i === this.selectedHeadIndex,
      });
      return html`<div class=${classes} @click=${handleClick}>${i}</div>`;
    };
    if (this.preds == null || this.preds[this.selectedLayer!] == null) {
      return;
    }
    const numHeadsPerLayer = this.preds![this.selectedLayer!].length;
    const numHeadsPerLayerRange =
        Array.from({length: numHeadsPerLayer}, (x: string, i: number) => i);
    // clang-format off
    return html`
        <label class="head-selector-label">Head:</label>
        ${numHeadsPerLayerRange.map(renderChip)}
    `;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, AttentionHeadsLitType);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'attention-module': AttentionModule;
  }
}
