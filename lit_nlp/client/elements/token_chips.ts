/**
 * @license
 * Copyright 2022 Google LLC
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

import './tooltip';

import {html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {styleMap} from 'lit/directives/style-map.js';

import {SalienceCmap, UnsignedSalienceCmap} from '../lib/colors';
import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './token_chips.css';

/**
 * A text token associated with a weight value
 */
export interface TokenWithWeight {
  token: string;
  weight: number;
  selected?: boolean;
  pinned?: boolean;
  onClick?: (e: Event) => void;
  onMouseover?: (e: Event) => void;
  onMouseout?: (e: Event) => void;
  disableHover?: boolean;
  forceShowTooltip?: boolean;
}

/**
 * An element that displays a sequence of tokens colored by their weights
 */
@customElement('lit-token-chips')
export class TokenChips extends LitElement {
  // List of tokens to display
  @property({type: Array}) tokensWithWeights: TokenWithWeight[] = [];
  @property({type: Object}) cmap: SalienceCmap = new UnsignedSalienceCmap();
  @property({type: String})
  tokenGroupTitle?: string;  // can be used for gradKey
  @property({type: Boolean}) dense = false;

  static override get styles() {
    return [sharedStyles, styles];
  }

  private renderToken(tokenInfo: TokenWithWeight, index: number) {
    const tokenClass = classMap({
      'salient-token': true,
      'selected': Boolean(tokenInfo.selected),
      'pinned': Boolean(tokenInfo.pinned),
      'clickable': tokenInfo.onClick !== undefined,
      'hover-enabled': !Boolean(tokenInfo.disableHover),
    });
    const tokenStyle = styleMap({
      'background-color': this.cmap.bgCmap(tokenInfo.weight),
      'color': this.cmap.textCmap(tokenInfo.weight),
    });

    // clang-format off
    return html`
      <div class=${tokenClass} style=${tokenStyle} @click=${tokenInfo.onClick}
        @mouseover=${tokenInfo.onMouseover} @mouseout=${tokenInfo.onMouseout}>
        <lit-tooltip content=${tokenInfo.weight.toPrecision(3)}
          ?forceShow=${Boolean(tokenInfo.forceShowTooltip)}
          ?disabled=${Boolean(tokenInfo.disableHover)}>
          <span class='pre-wrap' slot="tooltip-anchor">${tokenInfo.token}</span>
        </lit-tooltip>
      </div>`;
    // clang-format on
  }

  override render() {
    const tokensDOM = this.tokensWithWeights.map(
        (tokenWithWeight: TokenWithWeight, i: number) =>
            this.renderToken(tokenWithWeight, i));

    const holderClass = classMap({
      'tokens-holder': true,
      'tokens-holder-dense': this.dense,
    });

    // clang-format off
    return html`
      <div class="tokens-group">
        ${this.tokenGroupTitle ? this.tokenGroupTitle : ''}
        <div class=${holderClass}>
          ${tokensDOM}
        </div>
      </div>`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-token-chips': TokenChips;
  }
}
