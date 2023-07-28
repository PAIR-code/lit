/**
 * @license
 * Copyright 2023 Google LLC
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
 * @fileoverview A custom element that maps a LitType and (optionally) its
 * corresponding value to an HTML representation for editing or display.
 */

// tslint:disable:no-new-decorators

import {css, html} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {AnnotationCluster, EdgeLabel, ScoredTextCandidate, ScoredTextCandidates, SpanLabel} from '../lib/dtypes';
import {ReactiveElement} from '../lib/elements';
import {BooleanLitType, CategoryLabel, EdgeLabels, Embeddings, GeneratedTextCandidates, LitType, MultiSegmentAnnotations, Scalar, SpanLabels, StringLitType, TextSegment} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatAnnotationCluster, formatBoolean, formatEdgeLabel, formatEmbeddings, formatNumber, formatScoredTextCandidate, formatScoredTextCandidates, formatScoredTextCandidatesList, formatSpanLabel, renderCategoricalInput, renderTextInputLong, renderTextInputShort} from '../lib/type_rendering';
import {chunkWords} from '../lib/utils';

type LabelPlacement = 'left'|'right';

function isGeneratedTextCandidate(input: unknown): boolean {
  return Array.isArray(input) && input.length === 2 &&
         typeof input[0] === 'string' &&
         (input[1] == null || typeof input[1] === 'number');
}

/**
 * An element for mapping LitType fields to the correct HTML input type.
 *
 * **Usage**
 *
 * Editable mode:
 *
 *     <lit-input-field
 *       .type=${new Scalar()}
 *       .value=${1.234}
 *       @change=${hndlr}></lit-input-field>
 *
 * Readonly mode:
 *
 *     <lit-input-field readonly .value=${1.234}></lit-input-field>
 *
 * Flex to fill parent container:
 *
 *     <lit-input-field
 *       .type=${new Scalar()}
 *       .value=${1.234}
 *       @change=${hndlr}
 *       fill-container></lit-input-field>
 */
@customElement('lit-input-field')
export class LitInputField extends ReactiveElement {
  static override get styles () {
    return [sharedStyles, css`
      :host,
      :host > label {
        align-items: center;
        column-gap: 4px;
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
      }

      :host([fill-container]) > * {
        flex-grow: 1;
      }
    `];
  }

  @property({type: String}) label?: string;
  @property({type: String}) labelPlacement: LabelPlacement = 'left';
  @property({type: Object}) type = new LitType();
  @property({type: Boolean}) readonly = false;
  @property() value: unknown;
  @property({type: Boolean}) limitWords = false;

  override render() {
    const content = this.renderContent();
    if (this.label) {
      if (this.labelPlacement === 'left') {
        return html`<label>${this.label}${content}</label>`;
      } else {
        return html`<label>${content}${this.label}</label>`;
      }
    } else {
      return content;
    }
  }

  /**
   * Renders the appropriate string-like (readonly mode) or HTML (edit mode)
   * rpresentations of the field given its value and LitType.
   */
  private renderContent() {
    // Assignment to litType to avoid collision with TS reserved keyword 'type'.
    const litType = this.type;

    const booleanChange = (e: Event) => {
      this.value = (e.target as HTMLInputElement).checked;
      this.dispatchEvent(new Event('change'));
    };

    const numberChange = (e: Event) => {
      this.value = Number((e.target as HTMLInputElement).value);
      this.dispatchEvent(new Event('change'));
    };

    const stringChange = (e: Event) => {
      this.value = (e.target as HTMLSelectElement).value;
      this.dispatchEvent(new Event('change'));
    };

    if (litType instanceof BooleanLitType || typeof this.value === 'boolean') {
      return this.readonly ?
          formatBoolean(this.value as boolean) :
          html`<lit-checkbox ?checked=${this.value as boolean}
            @change=${booleanChange}></lit-checkbox>`;
    } else if (litType instanceof CategoryLabel) {
      // Always handle CategoryLabel before StringLitType because it is subclass
      if (this.readonly) {
        return this.formatAsString();
      } else if (litType.vocab != null) {
        return renderCategoricalInput(
          litType.vocab, stringChange, this.value as string
        );
      } else {
        return renderTextInputShort(stringChange, this.value as string);
      }
    } else if (litType instanceof EdgeLabels) {
      const formattedTags = (this.value as EdgeLabel[]).map(formatEdgeLabel);
      return formattedTags.join(', ');
    } else if (litType instanceof Embeddings) {
      return formatEmbeddings(this.value);
    } else if (litType instanceof GeneratedTextCandidates) {
      return formatScoredTextCandidatesList(
          this.value as ScoredTextCandidates[]);
    } else if (this.type instanceof MultiSegmentAnnotations) {
      const formattedTags =
          (this.value as AnnotationCluster[]).map(formatAnnotationCluster);
      return formattedTags.join(', ');
    } else if (litType instanceof Scalar || typeof this.value === 'number') {
      const {min_val, max_val, step} = litType as Scalar;
      return this.readonly ?
          formatNumber(this.value as number) :
          html`<lit-numeric-input .min=${min_val} .max=${max_val} .step=${step}
            value=${this.value as number} @change=${numberChange}>
          </lit-numeric-input>`;
    } else if (litType instanceof SpanLabels) {
      const formattedTags =
          (this.value as SpanLabel[]).map(s => formatSpanLabel(s)) as string[];
      return formattedTags.join(', ');
    } else if (litType instanceof TextSegment) {
      // Always handle TextSegment before StringLitType because it is subclass
      return this.readonly ?
          this.formatAsString() :
          renderTextInputLong(stringChange, this.value as string);
    } else if (litType instanceof StringLitType ||
               typeof this.value === 'string') {
      return this.readonly ?
          this.formatAsString() :
          renderTextInputShort(stringChange, this.value as string);

    } else if (Array.isArray(this.value)) {
      if (isGeneratedTextCandidate(this.value)) {
        return formatScoredTextCandidate(this.value as ScoredTextCandidate);
      }

      if (Array.isArray(this.value[0]) &&
          isGeneratedTextCandidate(this.value[0])) {
        return formatScoredTextCandidates(this.value as ScoredTextCandidates);
      }

      const strings = this.value.map((item) => {
        if (typeof item === 'number') {return formatNumber(item);}
        if (this.limitWords) {return chunkWords(item);}
        return `${item}`;
      });
      return strings.join(', ');
    } else {
      console.warn(
        `LitInputField does not support rendering '${litType}' fields.`
      );
      return '';
    }
  }

  private formatAsString() {
    return this.limitWords ? chunkWords(this.value as string) :
                             this.value as string;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-input-field': LitInputField;
  }
}
