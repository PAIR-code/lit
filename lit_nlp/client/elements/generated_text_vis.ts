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

// tslint:disable:no-new-decorators
import {html} from 'lit';
import {customElement, property} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {computed, observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {DiffMode, getTextDiff, TextDiff} from '../lib/generated_text_utils';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {ScoredTextCandidates} from '../lib/dtypes';

import {styles} from './generated_text_vis.css';

/** Generated text display, with optional diffs. */
@customElement('generated-text-vis')
export class GeneratedTextVis extends ReactiveElement {
  /* Data binding */
  @observable @property({type: String}) fieldName?: string;
  @observable
  @property({type: Array})
  candidates: ScoredTextCandidates = [];
  @observable @property({type: String}) referenceFieldName?: string;
  @observable @property({type: Array}) referenceTexts: ScoredTextCandidates = [];
  // Optional model scores for the target texts.
  @observable @property({type: Array}) referenceModelScores: number[] = [];

  @observable @property({type: String}) diffMode: DiffMode = DiffMode.NONE;
  @observable @property({type: Boolean}) highlightMatch: boolean = false;
  @observable @property({type: Number}) selectedIdx = 0;
  @observable @property({type: Number}) selectedRefIdx = 0;

  static override get styles() {
    return [sharedStyles, styles];
  }

  @computed
  get showModelScoreColumn() {
    return (
        this.referenceModelScores.length > 0 ||
        this.candidates.filter(candidate => candidate[1] != null).length > 0);
  }

  @computed
  get showTargetScoreColumn() {
    return this.referenceTexts.filter(candidate => candidate[1] != null)
               .length > 0;
  }

  @computed
  get showHeader() {
    return this.showModelScoreColumn || this.showTargetScoreColumn;
  }

  @computed
  get textDiff(): TextDiff|undefined {
    if (this.referenceTexts === undefined) return;
    if (this.candidates[this.selectedIdx] === undefined) return;
    if (this.diffMode === DiffMode.NONE) return;
    // Actually compute diffs - first get the selected output/reference.
    const outputText = this.candidates[this.selectedIdx][0];
    const byWord = (this.diffMode === DiffMode.WORD);
    const referenceText = this.referenceTexts[this.selectedRefIdx][0];
    return getTextDiff(referenceText, outputText, byWord);
  }

  renderDiffString(strings: string[], equal: boolean[]) {
    let displayStrings = strings;

    // Add spaces between strings for the word-wise character diffs.
    if (this.diffMode === DiffMode.WORD) {
      const lastIndex = strings.length - 1;
      displayStrings = strings.map((item, i) => {
        if (i !== lastIndex) {
          return item.concat(' ');
        }
        return item;
      });
    }

    const displaySpans = displayStrings.map((output, i) => {
      const classes = classMap({
        'highlighted-diff': !this.highlightMatch && !equal[i],
        'highlighted-match': this.highlightMatch && equal[i]
      });
      return html`<span class=${classes}>${output}</span>`;
    });
    return displaySpans;
  }

  renderHeader() {
    // clang-format off
    return html`
      <thead>
      <tr class='output-row'>
        <th class='field-name'>Field name</th>
        <td>
          <div class="candidate-row">
            ${this.showModelScoreColumn ? html`
              <div class='model-score column-header'>Model score</div>
            ` : null}
            ${this.showTargetScoreColumn ? html`
              <div class='reference-score column-header'>Target score</div>
            ` : null}
            <div class='candidate-text column-header'>Text</div>
          </div>
        </td>
      </tr>
      </thead>
    `;
    // clang-format on
  }

  renderReferences() {
    const renderedReferences = this.referenceTexts.map((candidate, i) => {
      const formattedText =
          this.textDiff !== undefined && i === this.selectedRefIdx ?
          this.renderDiffString(
              this.textDiff.inputStrings, this.textDiff.equal) :
              candidate[0];
      const classes = classMap({
        'token-chip-label': true,
        'candidate-row': true,
        'candidate-selected': this.selectedRefIdx === i,
      });
      const onClickSelect = () => {
        this.selectedRefIdx = i;
      };
      // clang-format off
      return html`
        <div class=${classes} @click=${onClickSelect}>
          ${this.showModelScoreColumn ? html`
            <div class='model-score' title="Model score">
              ${this.referenceModelScores[i]?.toFixed(3) ?? null}
            </div>
          `: null}
          ${this.showTargetScoreColumn ? html`
            <div class='reference-score' title="Reference score">
              ${candidate[1]?.toFixed(3) ?? null}
            </div>
          ` : null}
          <div class='candidate-text'>${formattedText}</div>
        </div>
      `;
      // clang-format on
    });

    // clang-format off
    return html`
      <tr class='output-row'>
        <th class='field-name'>${this.referenceFieldName}</th>
        <td><div class='candidates'>${renderedReferences}</div></td>
      </tr>
    `;
    // clang-format on
  }

  renderCandidates() {
    const renderedCandidates = this.candidates.map((candidate, i) => {
      const formattedText =
          this.textDiff !== undefined && i === this.selectedIdx ?
          this.renderDiffString(
              this.textDiff.outputStrings, this.textDiff.equal) :
          candidate[0];
      const classes = classMap({
        'token-chip-label': true,
        'candidate-row': true,
        'candidate-selected': this.selectedIdx === i,
      });
      const onClickSelect = () => {
        this.selectedIdx = i;
      };
      // clang-format off
      return html`
        <div class=${classes} @click=${onClickSelect}>
          ${this.showModelScoreColumn ? html`
            <div class='model-score'>
              ${candidate[1]?.toFixed(3) ?? null}
            </div>
          `: null}
          ${this.showTargetScoreColumn ? html`
            <div class='reference-score' title="Reference score">
            </div>
          ` : null}
          <div class='candidate-text'>${formattedText}</div>
        </div>
      `;
      // clang-format on
    });

    // clang-format off
    return html`
      <tr class='output-row'>
        <th class='field-name'>${this.fieldName}</th>
        <td><div class='candidates'>${renderedCandidates}</div></td>
      </tr>
    `;
    // clang-format on
  }

  override render() {
    // clang-format off
    return html`
      <div class='output'>
        <table class='output-table'>
          ${this.showHeader ? this.renderHeader() : null}
          <tbody>
            ${this.referenceFieldName != null ? this.renderReferences() : null}
            ${this.fieldName != null ? this.renderCandidates() : null}
          </tbody>
        </table>
      </div>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'generated-text-vis': GeneratedTextVis;
  }
}
