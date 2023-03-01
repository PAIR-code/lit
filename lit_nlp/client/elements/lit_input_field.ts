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

import {html} from 'lit';
import {customElement, property} from 'lit/decorators';
import {AnnotationCluster, EdgeLabel, ScoredTextCandidate, ScoredTextCandidates, SpanLabel} from '../lib/dtypes';
import {ReactiveElement} from '../lib/elements';
import {BooleanLitType, EdgeLabels, Embeddings, GeneratedTextCandidates, LitType, MultiSegmentAnnotations, Scalar, SpanLabels, StringLitType} from '../lib/lit_types';
import {formatAnnotationCluster, formatBoolean, formatEdgeLabel, formatEmbeddings, formatNumber, formatScoredTextCandidate, formatScoredTextCandidates, formatScoredTextCandidatesList, formatSpanLabel} from '../lib/type_rendering';
import {chunkWords} from '../lib/utils';

function isGeneratedTextCandidate(input: unknown): boolean {
  return Array.isArray(input) && input.length === 2 &&
         typeof input[0] === 'string' &&
         (input[1] == null || typeof input[1] === 'number');
}

/**
 * An element for mapping LitType fields to the correct HTML input type.
 */
@customElement('lit-input-field')
export class LitInputField extends ReactiveElement {
  @property({type: String}) name = '';
  @property({type: Object}) type = new LitType();
  @property({type: Boolean}) readonly = false;
  @property() value: unknown;
  @property({type: Boolean}) limitWords = false;

  override render() {
    return this.readonly ? this.renderReadonly() : this.renderEditable();
  }

  /**
   * Converts the input value to a readonly representaiton for display pruposes.
   */
  private renderReadonly(): string|number {
    if (Array.isArray(this.value)) {
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
    } else if (this.type instanceof BooleanLitType) {
      return formatBoolean(this.value as boolean);
    } else if (this.type instanceof EdgeLabels) {
      const formattedTags = (this.value as EdgeLabel[]).map(formatEdgeLabel);
      return formattedTags.join(', ');
    } else if (this.type instanceof Embeddings) {
      return formatEmbeddings(this.value);
    } else if (this.type instanceof GeneratedTextCandidates) {
      return formatScoredTextCandidatesList(
          this.value as ScoredTextCandidates[]);
    } else if (this.type instanceof MultiSegmentAnnotations) {
      const formattedTags =
          (this.value as AnnotationCluster[]).map(formatAnnotationCluster);
      return formattedTags.join(', ');
    } else if (this.type instanceof Scalar || typeof this.value === 'number') {
      return formatNumber(this.value as number);
    } else if (this.type instanceof SpanLabels) {
      const formattedTags =
          (this.value as SpanLabel[]).map(s => formatSpanLabel(s)) as string[];
      return formattedTags.join(', ');
    } else if (this.type instanceof StringLitType ||
               typeof this.value === 'string') {
      return this.limitWords ?
          chunkWords(this.value as string) : this.value as string;
    } else {
      return '';
    }
  }

  /**
   * Renders the appropriate HTML elements to enable user input to control the
   * value of the field with which this element is associated.
   */
  private renderEditable() {
    if (this.type instanceof BooleanLitType) {
      const change = (e: Event) => {
        this.value = !!(e.target as HTMLInputElement).value;
        dispatchEvent(new Event('change'));
      };
      return html`<lit-checkbox ?checked=${!!this.value} @change=${change}></lit-checkbox>`;
    } else {
      return html`Unsupported type '${this.type.name}' cannot be rnedered.`;
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-input-field': LitInputField;
  }
}
