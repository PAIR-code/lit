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
import {css, customElement, html, svg} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {IndexedInput, ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, getTextWidth, getTokOffsets, sumArray} from '../lib/utils';

import {styles as sharedStyles} from './shared_styles.css';

type Tokens = string[];
// <float>[num_heads, num_tokens, num_tokens]
type AttentionHeads = number[][][];

/**
 * A LIT module that renders the model's attention for a single input.
 */
@customElement('attention-module')
export class AttentionModule extends LitModule {
  static title = 'Attention';
  static numCols = 6;
  static duplicateForExampleComparison = true;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<attention-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></attention-module>`;
  };

  static get styles() {
    const styles = css`
        .head-selector-chip {
          margin: 0px 1px;
          width: 1rem;
          text-align: center;
        }

        .head-selector-chip.selected {
          color: #6403fa;
          border-color: #6403fa;
        }

        .head-selector-chip:hover {
          background: #f3e8fd;
        }
    `;
    return [sharedStyles, styles];
  }

  @observable private selectedLayer?: string;
  @observable private selectedHeadIndex: number = 0;
  @observable private preds?: {[key: string]: Tokens|AttentionHeads};

  firstUpdated() {
    const getSelectedInput = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getSelectedInput, selectedInput => {
      this.updateSelection(selectedInput);
    });
  }

  private async updateSelection(selectedInput: IndexedInput|null) {
    this.preds = undefined;  // clear previous results

    if (selectedInput === null) return;
    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [selectedInput], this.model, dataset, ['Tokens', 'AttentionHeads'],
        'Fetching attention');
    const res = await this.loadLatest('attentionAndTokens', promise);
    if (res === null) return;
    this.preds = res[0];
    // Make sure head selection is valid.
    const numHeadsPerLayer = this.preds[this.selectedLayer!] != null ?
        this.preds[this.selectedLayer!].length : 0;
    if (this.selectedHeadIndex >= numHeadsPerLayer) {
      this.selectedHeadIndex = 0;
    }
  }

  render() {
    if (!this.preds) return;

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
        <div class='module-results-area ${SCROLL_SYNC_CSS_CLASS}'>
          ${this.renderAttnHead()}
        </div>
      </div>
    `;
    // clang-format on
  }

  private renderAttnHead() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const fieldSpec = outputSpec[this.selectedLayer!];

    // Tokens involved in the attention.
    const inToks = (this.preds!)[fieldSpec.align_in!] as Tokens;
    const outToks = (this.preds!)[fieldSpec.align_out!] as Tokens;

    const fontFamily = "'Share Tech Mono', monospace";
    const fontSize = 12;
    const defaultCharWidth = 6.5;
    const font = `${fontSize}px ${fontFamily}`;
    const inTokWidths = inToks.map(tok => getTextWidth(tok, font, defaultCharWidth));
    const outTokWidths = outToks.map(tok => getTextWidth(tok, font, defaultCharWidth));
    const spaceWidth = getTextWidth(' ', font, defaultCharWidth);

    const inTokStr = svg`${inToks.join(' ')}`;
    const outTokStr = svg`${outToks.join(' ')}`;

    // Height of the attention visualization part.
    const visHeight = 100;

    // Vertical pad between attention vis and words.
    const pad = 10;

    // Calculate the full width and height.
    const inTokWidth = sumArray(inTokWidths) + (inToks.length - 1) * spaceWidth;
    const outTokWidth = sumArray(outTokWidths) + (outToks.length - 1) * spaceWidth;
    const width = Math.max(inTokWidth, outTokWidth);
    const height = visHeight + fontSize * 2 + pad * 4;

    // clang-format off
    return svg`
    <svg width=${width} height=${height}
      font-family="${fontFamily}"
      font-size="${fontSize}px">
      <text y=${pad * 2}> ${outTokStr}</text>
      ${this.renderAttnLines(visHeight, spaceWidth, 2.5 * pad, inTokWidths, outTokWidths)}
      <text y=${visHeight + 4 * pad}> ${inTokStr}</text>
    </svg>
    `;
    // clang-format on
  }

  /**
   * Render the actual lines between tokens to show the attention values.
   */
  private renderAttnLines(
      visHeight: number, spaceWidth: number, pad: number, inTokWidths: number[],
      outTokWidths: number[]) {
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
              return svg`
                <line
                  x1=${xOut(i)}
                  y1=${y1}
                  x2=${xIn(j)}
                  y2=${y2}
                  stroke="rgba(100,3,250,${attnVal})"
                  stroke-width=2>
                </line>`;
          })}`;
        });
    // clang-format on
  }

  /**
   * Render the dropdown with the layer names.
   */
  private renderLayerSelector() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const attnKeys = findSpecKeys(outputSpec, 'AttentionHeads');
    if (this.selectedLayer === undefined) {
      this.selectedLayer = attnKeys[0];
    }
    const onchange = (e: Event) => {
      this.selectedLayer = (e.target as HTMLSelectElement).value;
    };
    // clang-format off
    return html`
      <select class="dropdown" @change=${onchange}>
        ${attnKeys.map(key => html`<option value=${key}>${key}</option>`)}
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

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'AttentionHeads');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'attention-module': AttentionModule;
  }
}
