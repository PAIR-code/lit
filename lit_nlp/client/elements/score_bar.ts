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

import {property} from 'lit/decorators.js';
import {customElement} from 'lit/decorators.js';
import {css, html, LitElement} from 'lit';
import {styleMap, StyleInfo} from 'lit/directives/style-map.js';

const PRED_TITLE = 'The predicted label';
const TRUTH_TITLE = 'The ground truth label';

/**
 * A horizontal bar with a score label.
 */
@customElement('score-bar')
export class ScoreBar extends LitElement {
  // Score should be between 0 and maxScore.
  @property({type: Number}) score = 0;
  // Maximum score to be displayed by the score bar.
  @property({type: Number}) maxScore = 1;
  @property({type: Number}) scorePrecision = 3;
  @property({type: String}) barColor?: string;

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
    const barStyle: StyleInfo = {'width': `${scale * normalizedScore}%`};

    if (this.barColor) {
      barStyle['background-color'] = this.barColor;
    }

    return html`
        <div class='holder'>
          <div class='bar' style='${styleMap(barStyle)}'></div>
          <div class='text'>${this.score.toFixed(this.scorePrecision)}</div>
        </div>`;
  }
}

/**
 * A ScoreBar variant with annotations for the correct prediction and/or ground
 * truth value (if provided).
 */
@customElement('annotated-score-bar')
export class AnnotatedScoreBar extends LitElement {
  @property({type: Boolean}) hasTruth = false;
  @property({type: Boolean}) isPredicted = false;
  @property({type: Boolean}) isTruth = false;
  @property({type: Number}) value = 0;
  @property({type: String}) barColor?: string;

  static override get styles() {
    return css`
        .annotated-cell{
          display: flex;
          flex-direction: row;
          align-items: center;
          justify-content: space-between;
        }

        .indicator {
          min-width: 16px;
          text-align: center;
        }`;
  }

  override render() {
    return html`<div class="annotated-cell">
      <score-bar score=${this.value} maxScore=${1}
        .barColor=${this.barColor}></score-bar>
      <lit-tooltip content="${PRED_TITLE}">
        <div class="indicator" slot="tooltip-anchor">
          ${this.isPredicted ? 'P' : null}
        </div>
      </lit-tooltip>
      ${this.hasTruth ?
          html`<lit-tooltip content=${TRUTH_TITLE}>
          <div class="indicator" slot="tooltip-anchor">
            ${this.isTruth ? 'T' : null}
          </div></lit-tooltip>` : null}
    </div>`;
  }
}

/**
 * A legend for use with the AnnotatedScoreBar.
 */
@customElement('annotated-score-bar-legend')
export class AnnotatedScoreBarLegend extends LitElement {
  @property({type: Boolean}) hasTruth = false;

  override render() {
    return html`<span>
      P = ${PRED_TITLE}
      ${this.hasTruth ? `, T = ${TRUTH_TITLE}` : ''}
    </span>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'annotated-score-bar': AnnotatedScoreBar;
    'annotated-score-bar-legend': AnnotatedScoreBarLegend;
    'score-bar': ScoreBar;
  }
}
