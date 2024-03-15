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
  onClick?: (e: MouseEvent) => void;
  onMouseover?: (e: MouseEvent) => void;
  onMouseout?: (e: MouseEvent) => void;
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
  // Group title, such as the name of the active salience method.
  @property({type: String}) tokenGroupTitle?: string;
  /**
   * Dense mode, for less padding and smaller margins around each chip.
   */
  @property({type: Boolean}) dense = false;
  /**
   * breakNewlines removes \n at the beginning or end of a segment and inserts
   * explicit row break elements instead. Improves readability in many settings,
   * at the cost of "faithfulness" to the original token text.
   */
  @property({type: Boolean}) breakNewlines = false;
  /**
   * preSpace removes a leading space from a token and inserts an explicit
   * spacer element instead. Improves readability in many settings by giving
   * natural space between the highlight area for adjacent words, albeit at the
   * cost of hiding where the actual spaces are in the tokenization.
   */
  @property({type: Boolean}) preSpace = false;

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
      '--token-bg-color': this.cmap.bgCmap(tokenInfo.weight),
      '--token-text-color': this.cmap.textCmap(tokenInfo.weight),
    });

    let tokenText = tokenInfo.token;

    // TODO(b/324955623): render a gray '⏎' for newlines?
    // Maybe make this a toggleable option, as it can be distracting.
    // TODO(b/324955623): better rendering for multiple newlines, like \n\n\n ?
    // Consider adding an extra ' ' on each line.

    let preBreak = false;
    let postBreak = false;
    if (this.breakNewlines) {
      // Logic:
      // - \n : post-break, so blank space goes on previous line
      // - foo\n : post-break
      // - \nfoo : pre-break
      // - \n\n  : pre- and post-break, shows a space on its own line
      // - \n\n\n : pre- and post-break, two lines with only spaces
      if (tokenText.endsWith('\n')) {
        // Prefer post-break because this puts the blank space on the end of the
        // previous line, rather than creating an awkward indent on the next
        // one.
        tokenText = tokenText.slice(0, -1) + ' ';
        postBreak = true;
      }
      if (tokenText.startsWith('\n')) {
        // Pre-break only if \n precedes some other text.
        preBreak = true;
        tokenText = ' ' + tokenText.slice(1);
      }
    }

    let preSpace = false;
    if (this.preSpace && tokenText.startsWith(' ') && !preBreak) {
      preSpace = true;
      tokenText = tokenText.slice(1);
    }

    // Don't let token text shrink that much.
    if (tokenText === '') {
      tokenText = ' ';
    }

    // prettier-ignore
    return html`
      ${preSpace ? html`<div class='word-spacer'> </div>` : null}
      ${preBreak ? html`<div class='row-break'></div>` : null}
      <div class=${tokenClass} style=${tokenStyle} @click=${tokenInfo.onClick}
        @mouseover=${tokenInfo.onMouseover} @mouseout=${tokenInfo.onMouseout}>
        <lit-tooltip content=${tokenInfo.weight.toPrecision(3)}
          ?forceShow=${Boolean(tokenInfo.forceShowTooltip)}
          ?disabled=${Boolean(tokenInfo.disableHover)}>
          <span class='pre-wrap' slot="tooltip-anchor">${tokenText}</span>
        </lit-tooltip>
      </div>
      ${postBreak ? html`<div class='row-break'></div>` : null}
      `;
  }

  /**
   * Classes for the tokens holder. Subclass can add to this to enable
   * custom styling modes.
   */
  protected holderClass() {
    return {
      'tokens-holder': true,
      'dense': this.dense,
    };
  }

  override render() {
    const tokensDOM = this.tokensWithWeights.map(
        (tokenWithWeight: TokenWithWeight, i: number) =>
            this.renderToken(tokenWithWeight, i));

    // prettier-ignore
    return html`
      <div class="tokens-group">
        ${this.tokenGroupTitle ? this.tokenGroupTitle : ''}
        <div class=${classMap(this.holderClass())}>
          ${tokensDOM}
        </div>
      </div>`;
  }
}

/**
 * As above, but renders closer to running text - most notably by using
 * display: block and inline elements instead of flexbox.
 *
 * <lit-text-chips dense> should give text that is nearly indistinguishable
 * from that in a single div, while preserving all the click/hover/highlight
 * functionality of <lit-token-chips>.
 */
@customElement('lit-text-chips')
export class TextChips extends TokenChips {
  /**
   * Vertical dense mode, only affects vertical spacing.
   */
  @property({type: Boolean}) vDense = false;

  override holderClass() {
    return Object.assign(
        {}, super.holderClass(), {'text-chips': true, 'vdense': this.vDense});
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-token-chips': TokenChips;
    'lit-text-chips': TextChips;
  }
}
