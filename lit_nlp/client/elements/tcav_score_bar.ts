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

import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import {css, html, LitElement} from 'lit';
import {styleMap} from 'lit/directives/style-map';

/**
 * A custom wrapper around the mwc-checkbox which is a) smaller and b) has a
 * built-in label.
 */
@customElement('tcav-score-bar')
export class TcavScoreBar extends LitElement {
  @property({type: Number}) score = 0;
  @property({type: Number}) meanVal = 0;
  // maximum score value (any greater value gets clamped to this value)
  @property({type: Number}) clampVal = 0;

  static override get styles() {
    return css`
        .cell {
          position: relative;
          display: flex;
          min-width: 100px;
          min-height: 20px;
          background-color: var(--lit-neutral-200);
        }

        .separator {
          width: 2px;
          background-color: black;
          min-height: 20px;
          position: absolute;
        }

        .pos-bar {
          min-height: 20px;
          position: absolute;
          background-color: var(--lit-cyea-300);
        }
    `;
  }

  override render() {
    const {score, clampVal, meanVal} = this;
    const normalizedScore = Math.min(Math.max(score, 0), clampVal) / clampVal;
    const normalizedMean = Math.min(Math.max(meanVal, 0), clampVal) / clampVal;

    const stylePosBar: {[name: string]: string} = {
      'width': `${Math.abs(normalizedScore - normalizedMean) * 100}%`,
      'left': normalizedScore > normalizedMean ? `${normalizedMean * 100}%` :
                                                 `${normalizedScore * 100}%`
    };

    const styleSep: {[name: string]: string} = {
      'left': `${normalizedMean * 100}%`
    };

    return html`<td><div class='cell'>
        <div class='pos-bar' style='${styleMap(stylePosBar)}'></div>
        <div class='separator' style='${styleMap(styleSep)}'></div>
      </div></td>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'tcav-score-bar': TcavScoreBar;
  }
}
