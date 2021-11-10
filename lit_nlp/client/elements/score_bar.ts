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
 * A horizontal bar with a score label.
 */
@customElement('score-bar')
export class ScoreBar extends LitElement {
  // Score should be between 0 and maxScore.
  @property({type: Number}) score = 0;
  // Maximum score to be displayed by the score bar.
  @property({type: Number}) maxScore = 1;

  static override get styles() {
    return css`
        .holder {
          width: 100px;
          height: 20px;
          display: flex;
          position: relative;
        }

        .bar {
          padding-left: .75%;
          padding-right: .75%;
          margin-left: .35%;
          margin-right: .35%;
          background-color: var(--lit-cyea-300);
        }

        .text {
          position: absolute;
          padding-left: 4px;
          padding-right: 2px;
          color: var(--lit-neutral-800);
        }
    `;
  }

  override render() {
    const normalizedScore = this.score / this.maxScore;
    const scale = 90;
    const barStyle = {'width': `${scale * normalizedScore}%`};
    return html`
        <div class='holder'>
          <div class='bar' style='${styleMap(barStyle)}'></div>
          <div class='text'>${this.score.toFixed(3)}</div>
        </div>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'score-bar': ScoreBar;
  }
}
