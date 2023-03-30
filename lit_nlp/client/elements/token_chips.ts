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

import {html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators';
import {styleMap} from 'lit/directives/style-map';

import {SalienceCmap, UnsignedSalienceCmap} from '../lib/colors';
import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './token_chips.css';

/**
 * A text token associated with a weight value
 */
export interface TokenWithWeight {
  token: string;
  weight: number;
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

  static override get styles() {
    return [sharedStyles, styles];
  }

  private renderToken(token: string, weight: number, index: number) {
    const tokenStyle = styleMap({
      'background-color': this.cmap.bgCmap(weight),
      'color': this.cmap.textCmap(weight),
    });

    // clang-format off
    return html`
      <div class="salient-token" style=${tokenStyle}>
        <lit-tooltip content=${weight.toPrecision(3)}>
          <span>${token}</span>
        </lit-tooltip>
      </div>`;
    // clang-format on
  }

  override render() {
    const tokensDOM = this.tokensWithWeights.map(
        (tokenWithWeight: TokenWithWeight, i: number) =>
            this.renderToken(tokenWithWeight.token, tokenWithWeight.weight, i));

    // clang-format off
    return html`
      <div class="tokens-group">
        ${this.tokenGroupTitle ? this.tokenGroupTitle : ''}
        <div class="tokens-holder">
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
