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
 * @fileoverview Functions for converting LitType classes to base HTML elements
 * for editing or display.
 *
 * As a naming convention, exported functions that support editing via HTML
 * inputs, selects, etc, start with "render" (because they used Lit.dev's HTML
 * templates to render elements) and functions that are intended for readonly
 * display start with "format" (because they typically convert the value to a
 * string-like representation).
 *
 * Default styles for elements returned by this function are defined in
 * ./shared_styles.css. Consumers of this library should import that stylesheet
 * and return it as part of their styles getter.
 */

import {html} from 'lit';
import {StyleInfo, styleMap} from 'lit/directives/style-map.js';
import {AnnotationCluster, EdgeLabel, ScoredTextCandidate, ScoredTextCandidates, SpanLabel} from './dtypes';

type EventHandler = (e: Event) => void;

/** Formats an AnnotationCluster for display. */
export function formatAnnotationCluster(ac: AnnotationCluster): string {
  return `${ac.label}${ac.score != null ? ` (${ac.score})` : ''}`;
}

/**
 * Formats an AnnotationCluster[] (e.g., from MultiSegmentAnnotations) for
 * display as a <div> of <div>s.
 */
export function formatAnnotationClusters(clusters: AnnotationCluster[]) {
  return html`<div class="multi-segment-annotation">${clusters.map(c =>
    html`<div class="annotation-cluster">
      <div>${formatAnnotationCluster(c)}</div>
      <ul>${c.spans.map(s => html`<li>${formatSpanLabel(s, true)}</li>`)}</ul>
    </div>`)
  }</div>`;
}

/** Formats a boolean value for display. */
export function formatBoolean(val: boolean): string {
  return val ? '✔' : ' ';
}

/** Formats an EdgeLabel for display. */
export function formatEdgeLabel(e: EdgeLabel): string {
  function formatSpan (s: [number, number]) {return `[${s[0]}, ${s[1]})`;}
  const span1Text = formatSpan(e.span1);
  const span2Text = e.span2 ? ` ← ${formatSpan(e.span2)}` : '';
  // Add non-breaking control chars to keep this on one line
  // TODO(lit-dev): get it to stop breaking between ) and :; \u2060 doesn't work
  return `${span1Text}${span2Text}\u2060: ${e.label}`.replace(
      /\ /g, '\u00a0' /* &nbsp; */);
}

/** Formats Embeddings for display. */
export function formatEmbeddings(val: unknown): string {
  return Array.isArray(val) ? `<float>[${val.length}]` : '';
}

/** Formats a number to a fixed (3 decimal) value for display. */
export function formatNumber (item: number): number {
  return Number.isInteger(item) ? item : Number(item.toFixed(3));
}

/** Formats a ScoredTextCandidate for display. */
export function formatScoredTextCandidate([t, s]: ScoredTextCandidate): string {
  return `${t}${typeof s === 'number' ? ` (${formatNumber(s)})` : ''}`;
}

/** Formats a ScoredTextCandidates for display. */
export function formatScoredTextCandidates(stc: ScoredTextCandidates): string {
  return stc.map(formatScoredTextCandidate).join('\n\n');
}

/** Formats a list of ScoredTextCandidates for display. */
export function formatScoredTextCandidatesList(
    list: ScoredTextCandidates[]): string {
  return list.map(formatScoredTextCandidates).join('\n\n');
}

/** Formats a SpanLabel for display.
 *
 * By default, it formats the SpanLabel as a string. If monospace is true, then
 * the function returns an HTML Template wrapping the default string in a
 * <div.monospace-label> element.
 */
export function formatSpanLabel(s: SpanLabel, monospace = false) {
  let formatted = `[${s.start}, ${s.end})`;
  if (s.align) {
    formatted = `${s.align} ${formatted}`;
  }
  if (s.label) {
    // TODO(lit-dev): Stop from breaking between ) and :, \u2060 doesn't work
    formatted = `${formatted}\u2060: ${s.label}`;
  }

  formatted = formatted.replace(/\ /g, '\u00a0' /* &nbsp; */);
  return monospace ?
      html`<div class="monospace-label">${formatted}</div>` : formatted;
}

/** Formats a SpanLabel[] as a <div> of <div>s. */
export function formatSpanLabels(labels: SpanLabel[]) {
  return html`<div class="span-labels">${
    labels.map(s =>formatSpanLabel(s, true))
  }</div>`;
}

/** Renders a <select> element with <option>s for each item in the vocab. */
export function renderCategoricalInput(
    vocab: string[], change: EventHandler, value = '') {
  return html`<select class="dropdown" @change=${change} .value=${value}>
    ${vocab.map(cat =>
        html`<option value=${cat} ?selected=${value === cat}>${cat}</option>`)}
    <option value="" ?selected=${value === ``}></option>
  </select>`;
}

/** Renders a <textarea> for long-form textual input. */
export function renderTextInputLong(
    input: EventHandler, value = '', styles: StyleInfo = {}) {
  return html`<textarea class="input-box" style="${styleMap(styles)}"
      @input=${input} .value=${value}></textarea>`;
}

/** Renders an <inut type="text"> for short-form textual input. */
export function renderTextInputShort(input: EventHandler, value = '') {
  return html`<input type="text" class="input-short" @input=${input}
      .value=${value}>`;
}
