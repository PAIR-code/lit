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

import {css, customElement, html, LitElement, property} from 'lit-element';
import {styleMap} from 'lit-html/directives/style-map';

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

  static get styles() {
    return css`
        .cell {
          position: relative;
          display: flex;
          min-width: 100px;
        }

        .separator {
          width: 2px;
          background-color: black;
          min-height: 20px;
          position: absolute;
        }

        .pos-bar {
          background-color: #4ECDE6;
          min-height: 20px;
        }

        .pos-blank {
          background-color: #E8EAED;
          min-height: 20px;
        }
    `;
  }

  render() {
    const score = this.score;
    const clampVal = this.clampVal;
    const meanVal = this.meanVal;

    const stylePosBlank: {[name: string]: string} = {};
    stylePosBlank['width'] =
        `${(1 - Math.min(Math.max(score, 0), clampVal) / clampVal) * 100}%`;

    const stylePosBar: {[name: string]: string} = {};
    stylePosBar['width'] =
        `${Math.min(Math.max(score, 0), clampVal) / clampVal * 100}%`;

    const styleSep: {[name: string]: string} = {};
    styleSep['left'] =
        `${Math.min(Math.max(meanVal, 0), clampVal) / clampVal * 100}%`;

    return html`<td><div class='cell'>
        <div class='separator' style='${styleMap(styleSep)}'></div>
        <div class='pos-bar' style='${styleMap(stylePosBar)}'></div>
        <div class='pos-blank' style='${styleMap(stylePosBlank)}'></div>
        </div></td>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'tcav-score-bar': TcavScoreBar;
  }
}
