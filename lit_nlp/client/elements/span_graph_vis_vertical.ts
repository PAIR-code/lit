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

/**
 * Visualization component for structured prediction over text.
 */

// tslint:disable:no-new-decorators

import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {observable} from 'mobx';

import {getVizColor} from '../lib/colors';
import {EdgeLabel} from '../lib/dtypes';
import {ReactiveElement} from '../lib/elements';

import {styles} from './span_graph_vis_vertical.css';


/**
 * Represents a group of directed graphs anchored to token spans.
 * This is the general "edge probing" representation, which can be used for many
 * problems including sequence tagging, span labeling, and directed graphs like
 * semantic frames, coreference, and dependency parsing. See
 * https://arxiv.org/abs/1905.06316 for more on this formalism.
 */
export interface SpanGraph {
  tokens: string[];
  layers: AnnotationLayer[];
}

/**
 * A single layer of annotations, like 'pos' (part-of-speech)
 * or 'ner' (named entities).
 */
export interface AnnotationLayer {
  name: string;
  edges: EdgeLabel[];
  hideBracket?: boolean;
}

function formatEdgeLabel(label: string|number): string {
  if (typeof (label) === 'number') {
    return Number.isInteger(label) ? label.toString() :
                                     label.toFixed(3).toString();
  }
  return `${label}`;
}

/** Structured prediction (SpanGraph) visualization class. */
@customElement('span-graph-vis-vertical')
export class SpanGraphVis extends ReactiveElement {
  /* Data binding */
  @property({ type: Object }) data: SpanGraph = { tokens: [], layers: [] };
  @property({ type: Boolean }) showLayerLabel: boolean = true;

  @observable private selectedTokIdx?: number;
  @observable private readonly columnVisibility: { [key: string]: boolean } = {};

  /* Rendering parameters */
  @property({ type: Number }) lineHeight: number = 18;
  @property({ type: Number }) approxFontSize = this.lineHeight / 3;

  // Padding for SVG viewport, to avoid clipping some elements (like polyline).
  @property({ type: Number }) viewPad: number = 5;

  static override get styles() {
    return styles;
  }

  override render() {
    if (!this.data) {
      return ``;
    }
    const host = this.shadowRoot!.host as HTMLElement;
    host.style.setProperty('--line-height', `${this.lineHeight}pt`);
    const tokens = this.data.tokens;

    const tokenClasses = (i: number) => classMap({
      line: true,
      token: true,
      selected: i === this.selectedTokIdx
    });

    // clang-format off
    return html`
    <div class='holder' >
      <div class='background-lines'>
        ${tokens.map(t => html`<div class=background-line></div>`)}
      </div>
      <div class='tokens'>
          ${tokens.map((t, i) => html`
          <div class=${tokenClasses(i)}
            @mouseenter=${() => this.selectedTokIdx = i}
            @mouseleave=${() => this.selectedTokIdx = undefined}>
            ${t}
          </div>
          `)}
      </div>
      ${this.data.layers.map((layer, i) => this.renderLayer(layer, i))}
    </div>`;
    // clang-format on
  }

  /**
   * Render a given annotation layer.
   */
  renderLayer(layer: AnnotationLayer, i: number) {

    if (!layer.edges.length) {
      return html``;
    }

    const layerStyles = styleMap({
      '--group-color': getVizColor('dark', i).color
    });

    // The column width is the width of the longest label, in pixels.
    const colWidth =
        Math.max(
            layer.name.length,
            ...layer.edges.map(e => formatEdgeLabel(e.label).length)) *
            this.approxFontSize +
        this.viewPad * 2;

    const colStyles = styleMap({ width: `${colWidth}pt` });
    const hidden = this.columnVisibility[layer.name];
    const columnClasses = classMap({
      'column': true,
      'hidden': hidden
    });

    const headerClasses = classMap({ 'layer-label-vert': true, hidden });
    const onClick = () =>
      this.columnVisibility[layer.name] = !this.columnVisibility[layer.name];

    // clang-format off
    return html`
    <div class=layer style=${layerStyles} @click=${onClick}>
      ${this.showLayerLabel ? html`
      <div class=${headerClasses}>
        ${layer.name}
      </div>` : null}
      <div style=${colStyles} class=${columnClasses}>
        ${layer.edges.map(edge => this.renderEdge(edge, layer, colWidth))}
      </div>
    </div>
    `;
    // clang-format on
  }

  /**
   * Render an edge and its label. See the note on the SpanGraph interface
   * above for more details.
   */
  private renderEdge(edge: EdgeLabel, layer: AnnotationLayer, colWidth: number) {
    const isArc = 'span2' in edge;
    const span0 = edge.span1[0];
    const span1 = edge.span2 ? edge.span2[0] : edge.span1[1];
    const topSpan = Math.min(span0, span1);
    const botSpan = Math.max(span0, span1);


    const isInSpan = (i: number, span:[number, number]) => i >= span[0] && i < span[1];

    // Span classes (child, parent, etc, based on the currently selected token.)
    const tokSelected = this.selectedTokIdx !== undefined;
    const selected = isInSpan(this.selectedTokIdx!, edge.span1) || (isArc && isInSpan(this.selectedTokIdx!, edge.span2!));
    const child = isArc && this.selectedTokIdx === span1;
    const parent = isArc && this.isChildOfSelected(layer, span0);
    const grayLine = tokSelected && !(selected || child);
    const grayLabel = grayLine && !(parent);

    // Edge labels can be either strings or numbers; format the latter nicely.
    const formattedLabel = formatEdgeLabel(edge.label);

    // Styling for the label text.
    const labelWidthInPx = formattedLabel.length * this.approxFontSize;
    const labelStyle = styleMap({
      top: `${span0 * this.lineHeight}pt`,
      left: isArc ? `${colWidth - labelWidthInPx - this.viewPad}pt` : '',
    });
    const labelClasses = classMap({
      child, parent, selected,
      gray: grayLabel,
      line: true,
      edge: true
    });

    // Styling for the arc (a line and sometimes an arrowhead)
    const arcPad = .3;
    const offset = this.lineHeight / 8;
    const top = isArc ?
      (topSpan + arcPad) * this.lineHeight + (topSpan === span0 ? 0 : this.viewPad) :
      topSpan * (this.lineHeight) - offset;
    const bottom = isArc ?
      (botSpan + arcPad) * this.lineHeight + (botSpan === span0 ? 0 : -this.viewPad) :
      botSpan * (this.lineHeight) - 2 * offset;

    const arcHeight = bottom - top;
    const width = isArc ? `${Math.max(arcHeight / 2, this.lineHeight / 2)}pt` : '';

    const rad = isArc ? arcHeight / 2 : 3;
    const lineStyle = styleMap({
      top: `${top}pt`,
      height: `${arcHeight}pt`,
      width,
      'border-radius': `0pt ${rad}pt ${rad}pt 0pt`,
      left: isArc ? `${colWidth + 10}pt` : '',
      visibility: layer.hideBracket ? 'hidden' : 'visble',
    });

    const arrowHeadClasses = classMap({
      'arrow-head': true,
      'bottom': topSpan === span1,
    });

    const arrowClasses = classMap({
      child,
      parent: selected,
      gray: grayLine,
      edge: true,
      'edge-line': true
    });

    return html`
    <div style=${lineStyle} class='${arrowClasses}'>
      ${isArc ? html`<div class=${arrowHeadClasses}></div>` : ''}
    </div>
    <div class=${labelClasses}
      style=${labelStyle}>
      ${formattedLabel}
    </div>
    `;
  }

  /**
   * Is this token (indicated by tokenIdx) a child of the selected token at
   * the specified layer. This assumes that the edge goes from span1 to span2,
   * as in a dependency parse tree.
   */
  isChildOfSelected(layer: AnnotationLayer, tokenIdx: number) {
    for (let j = 0; j < layer.edges.length; j++) {
      const edge = layer.edges[j];
      if (edge.span2 &&
        (this.selectedTokIdx === edge.span1[0]) &&
        (tokenIdx === edge.span2[0])) {
        return true;
      }
    }
    return false;
  }

}

declare global {
  interface HTMLElementTagNameMap {
    'span-graph-vis-vertical': SpanGraphVis;
  }
}
