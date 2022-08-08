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

import * as d3 from 'd3';
import {html, LitElement, svg} from 'lit';
import {customElement, property} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';

import {getVizColor} from '../lib/colors';
import {EdgeLabel} from '../lib/dtypes';

import {styles} from './span_graph_vis.css';


/**
 * Represents a group of directed graphs anchored to token spans.
 * This is the general "edge probing" representation, which can be used for many
 * problems including sequence tagging, span labeling, and directed graphs like
 * semantic frames, coreference, and dependency parsing. See
 * https://arxiv.org/abs/1905.06316 for more on this formalism.
 */
export interface SpanGraph {
  'tokens': string[];
  'layers': AnnotationLayer[];
}

/**
 * A single layer of annotations, like 'pos' (part-of-speech)
 * or 'ner' (named entities).
 */
export interface AnnotationLayer {
  'name': string;
  'edges': EdgeLabel[];
}

/* Compute points for a polyline bracket. */
function hBracketPoints(width: number, height: number, lean: number) {
  // Points for a polyline bracket.
  const start = [0, 0];
  const ltop = [lean, height];
  const rtop = [width - lean, height];
  const end = [width, 0];
  return [start, ltop, rtop, end];
}

/**
 * Compute path for a dependency arc.
 */
function arcPath(
    startY: number, x1: number, x2: number, height: number, aspect: number) {
  const left = Math.min(x1, x2);
  const right = Math.max(x1, x2);
  let pathCommands = `M ${left} ${startY} `;
  if ((right - left) > (2 * aspect * height)) {
    // Long arcs: draw as 90* curve, flat, then 90* curve.
    const majorAxis = aspect * height;
    pathCommands += `A ${majorAxis} ${height} 0 0 1 ${left + majorAxis} ${
        startY - height} `;
    pathCommands += `L ${right - majorAxis} ${startY - height} `;
    pathCommands += `A ${majorAxis} ${height} 0 0 1 ${right} ${startY} `;
  } else {
    // Short arcs: draw as single 180* curve.
    height = (right - left) / (2 * aspect);
    pathCommands +=
        `A ${(right - left) / 2} ${height} 0 0 1 ${right} ${startY} `;
  }
  return pathCommands; /* assign as 'd' attribute to path */
}

/**
 * Compute path for the arrow at the end of an arc.
 */
function arcArrow(startY: number, x: number, markSize: number) {
  let pathCommands = `M ${x - markSize} ${startY - (1.5 * markSize)} `;
  pathCommands += `L ${x + markSize} ${startY - (1.5 * markSize)} `;
  pathCommands += `L ${x} ${startY} Z`;
  return pathCommands; /* assign as 'd' attribute to path */
}

/* Set attributes to match target's size to the source element. */
function matchBBox(source: SVGGElement, target: SVGRectElement) {
  const bbox = source.getBBox();
  target.setAttribute('x', `${bbox.x}`);
  target.setAttribute('y', `${bbox.y}`);
  target.setAttribute('width', `${bbox.width}`);
  target.setAttribute('height', `${bbox.height}`);
}

/** Structured prediction (SpanGraph) visualization class. */
@customElement('span-graph-vis')
export class SpanGraphVis extends LitElement {
  /* Data binding */
  @property({type: Object}) data: SpanGraph = {tokens: [], layers: []};
  @property({type: Boolean}) showLayerLabel: boolean = true;

  /* Rendering parameters */
  @property({type: Number}) lineHeight: number = 18;
  @property({type: Number}) bracketHeight: number = 5.5;
  @property({type: Number}) yPad: number = 5;
  // For arcs between spans.
  @property({type: Number}) arcBaseHeight: number = 20;
  @property({type: Number}) arcMaxHeight: number = 40;
  @property({type: Number}) arcAspect: number = 1.2;
  @property({type: Number}) arcArrowSize: number = 4;
  // Padding for SVG viewport, to avoid clipping some elements (like polyline).
  @property({type: Number}) viewPad: number = 5;
  // Multiplier from SVG units to screen pixels.
  @property({type: Number}) svgScaling: number = 1.2;

  /* Internal rendering state */
  private tokenXBounds: Array<[number, number]> = [];

  static override get styles() {
    return styles;
  }

  renderTokens(tokens: string[]) {
    return svg`
      <g id='token-group'>
        <text class='token-text' height=${this.lineHeight}>
          ${tokens.map(t => svg`<tspan>${svg`${t + ' '}`}</tspan>`)}
        </text>
      </g>`;
  }

  private getTokenGroup() {
    return this.shadowRoot!.querySelector('g#token-group') as SVGGElement;
  }

  renderEdge(edge: EdgeLabel, color: string) {
    // Positioning relative to the group transform, which will be applied later.
    const labelHeight = this.lineHeight;
    const labelY = -(this.bracketHeight + this.lineHeight);

    let labelText = edge.label;
    let isNegativeEdge = false;
    if (typeof edge.label === 'number') {
      labelText = edge.label.toFixed(3);
      isNegativeEdge = edge.label < 0.5;
    }
    const arcPathClass =
        classMap({'arc-path': true, 'arc-neg': isNegativeEdge});
    const arcArrowClass =
        classMap({'arc-arrow': true, 'arc-neg': isNegativeEdge});
    color = isNegativeEdge ? 'gray' : color;
    // clang-format off
    return svg`
      <g class='edge-group' style=${styleMap({'--group-color': color})}
        data-color=${color}>
        ${edge.span2 ? svg`
          <path class=${arcPathClass}></path>
          <path class=${arcArrowClass}></path>
          ` : ''}
        <g class='at-span1'>
          <polyline class='span-bracket'></polyline>
          <foreignObject class='span-label' height=${labelHeight} y=${labelY}>
            ${html`<div>${labelText}</div>`}
          </foreignObject>
          <rect class='mousebox'></rect>
        </g>
        ${edge.span2 ? svg`<g class='at-span2'>
            <polyline class='span-bracket'></polyline>
          </g>` : ''}
      </g>
    `;
    // clang-format on
  }

  renderLayer(layer: AnnotationLayer, i: number) {
    const rowColor = getVizColor('deep', i).color;
    // Positioning relative to the group transform, which will be applied later.
    const rowLabelX = -10;
    const rowLabelY = -(this.bracketHeight + 0.5 * this.lineHeight);

    const orderedEdges = this.sortEdges(layer.edges);
    // clang-format off
    return svg`
      <g id='layer-group-${layer.name}' data-color=${rowColor}>
        ${this.showLayerLabel ? svg`
          <g class='layer-label'>
            <text x=${rowLabelX} y=${rowLabelY} fill=${rowColor}>
              ${svg`${layer.name}`}
            </text>
          </g>` : null}
        ${orderedEdges.map(edge => this.renderEdge(edge, rowColor))}
      </g>
    `;
    // clang-format on
  }

  private getLayerGroup(name: string) {
    return this.shadowRoot!.querySelector(`g#layer-group-${name}`) as
        SVGGElement;
  }

  override render() {
    return svg`
      <svg id='svg' xmlns='http://www.w3.org/2000/svg'><g id='all'>
        ${this.data ? this.renderTokens(this.data.tokens) : ''}
        ${this.data ? this.data.layers.map(this.renderLayer.bind(this)) : ''}
      </g></svg>`;
  }

  private findTokenBounds() {
    const tokenNodes = this.getTokenGroup().querySelectorAll('tspan');
    const tokenXBounds: Array<[number, number]> = [];
    tokenNodes.forEach(tspan => {
      // Use getBBox() to avoid a crash when tspan.getNumberOfChars() === 0.
      // TODO(lit-dev): figure out why this case happens - maybe
      // the nodes are not yet attached to the DOM?
      const bbox = tspan.getBBox();
      tokenXBounds.push([bbox.x, bbox.x + bbox.width]);
    });
    return tokenXBounds;
  }

  /**
   * Consistent sort order.
   * Because span labels overflow to the right, we order these so the rightmost
   * spans appear first in the DOM, and thus render under anything to the left
   * that needs to overflow.
   */
  private sortEdges(edges: EdgeLabel[]) {
    return edges.slice().sort((a, b) => d3.descending(a.span1[1], b.span1[1]));
  }

  /* Starting x position for a bracket, in SVG coordinates */
  private getStartX(span: [number, number]) {
    return this.tokenXBounds[span[0]][0];
  }

  /* Ending x position for a bracket, in SVG coordinates */
  private getEndX(span: [number, number]) {
    return this.tokenXBounds[span[1] - 1][1];
  }

  /* Find available width without clipping the next label */
  private findAvailableWidths(layerGroup: Element, edges: EdgeLabel[]):
      number[] {
    const availableWidths: number[] = edges.map(() => 0);
    // Find available space for each label, by checking where the next label
    // starts. We iterate from right to left through the spans, starting with
    // the second-rightmost (i=1).
    for (let i = 1; i < edges.length; i++) {
      const edge = edges[i];          // this span
      const nextEdge = edges[i - 1];  // right neighboring span
      availableWidths[i] =
          this.getStartX(nextEdge.span1) - this.getStartX(edge.span1);
    }
    // We don't want the rightmost label (index 0) to be cut off by the edge
    // of the SVG draw area, even if the label extends past the end of the
    // text. So we need to:
    // 1) Set this label to fit the content, so the bounding box contains all
    // the label text.
    // 2) Set the available width to this rendered width, so we don't clip it
    // later.
    const firstSpanDiv =
        // tslint:disable-next-line:no-unnecessary-type-assertion
        layerGroup.querySelector('g.edge-group foreignObject div') as
            HTMLDivElement |
        null;
    if (firstSpanDiv !== null) {
      firstSpanDiv.style.width = 'fit-content';
      availableWidths[0] = firstSpanDiv.getBoundingClientRect().width;
    }
    return availableWidths;
  }

  /* Set mouseovers, using d3. */
  private setMouseovers(group: SVGGElement, edges: EdgeLabel[]) {
    const rowColor = group.dataset['color'] as string;
    const grayColor = getVizColor('deep', 'other').color;

    const spanGroups = d3.select(group).selectAll('g.edge-group').data(edges);
    const tokenSpans = d3.select(this.getTokenGroup()).selectAll('tspan');

    // On mouseover, highlight this span and the corresponding text.
    spanGroups.each(function(d, i) {
      const colorFn = (e: unknown, j: number) =>
          (i === j) ? rowColor : grayColor;
      const tokenColorFn = (t: unknown, j: number) => {
        const inSpan1 = (d.span1[0] <= j && j < d.span1[1]);
        const inSpan2 = d.span2 ? (d.span2[0] <= j && j < d.span2[1]) : false;
        return (inSpan1 || inSpan2) ? rowColor : 'black';
      };
      const mouseBox = d3.select(this).select('rect.mousebox');
      mouseBox.on('mouseover', () => {
        spanGroups.style('--group-color', colorFn);
        tokenSpans.attr('fill', tokenColorFn);
        d3.select(this).classed('selected', true);
        // Ideally we'd also move this element so that it renders above
        // the other groups, but SVG2 z-index is not supported by most browsers
        // and simply reordering child nodes does not play well with lit-html's
        // rendering logic, which relies on pointers to specific positions in
        // the DOM.
        // d3.select(this).classed('selected', true).raise();
        // TODO(iftenney): consider implementing a tooltip that clones this
        // element but always renders above the other spans.
      });
      mouseBox.on('mouseout', () => {
        // Reset to original color, stored on group element.
        // TODO(lit-dev): do this with another CSS class instead?
        spanGroups.style('--group-color', function(e) {
          return (this as SVGElement).dataset['color'] as string;
        });
        tokenSpans.attr('fill', 'black');
        d3.select(this).classed('selected', false);
      });
    });
  }

  /* Set y-position of rendered layers */
  private positionLayers() {
    let rowStartY = this.getTokenGroup().getBBox().y - this.yPad / 2;
    for (let i = 0; i < this.data.layers.length; i++) {
      const group: SVGGElement = this.getLayerGroup(this.data.layers[i].name);
      group.setAttribute('transform', `translate(0, ${rowStartY})`);
      rowStartY -= group.getBBox().height + this.yPad;
    }
  }

  /* Set the SVG viewport to the bounding box of the main group. */
  private setSVGViewport() {
    const mainGroup = this.shadowRoot!.querySelector('g#all') as SVGGElement;
    const bbox = mainGroup.getBBox();
    const svg = this.shadowRoot!.getElementById('svg')!;
    // Set bounding box to cover main group + viewPad on all sides.
    const viewBox = [
      bbox.x - this.viewPad, bbox.y - this.viewPad,
      bbox.width + 2 * this.viewPad, bbox.height + 2 * this.viewPad
    ];
    svg.setAttribute('viewBox', `${viewBox}`);
    // Set the height of the SVG as it will render on the page.
    svg.setAttribute(
        'height', `${this.svgScaling * (bbox.height + 2 * this.viewPad)}`);
  }

  /**
   * Post-render callback. Performs imperative updates to layout and component
   * sizes which need to depend on the positions of each token. Also sets up
   * mouseover behavior.
   */
  override updated() {
    if (this.data == null) {
      this.tokenXBounds = [];
      return;
    }
    this.tokenXBounds = this.findTokenBounds();

    // For each layer, position the span groups
    for (const layer of this.data.layers) {
      const orderedEdges = this.sortEdges(layer.edges);

      // Container group for this layer.
      const layerGroup: SVGGElement = this.getLayerGroup(layer.name);

      // Compute available widths, needed for clipping of labels.
      const availableWidths =
          this.findAvailableWidths(layerGroup, orderedEdges);

      // Edge groups within this layer.
      const edgeGroups = layerGroup.querySelectorAll('g.edge-group');
      edgeGroups.forEach((g, i) => {
        const edge = orderedEdges[i];

        const g1 = g.querySelector('g.at-span1')!;
        // Set position within this row.
        g1.setAttribute(
            'transform', `translate(${this.getStartX(edge.span1)}, 0)`);

        // Compute span width in SVG units, based on rendered token width.
        const span1Width =
            this.getEndX(edge.span1) - this.getStartX(edge.span1);
        // Set points for span1 bracket.
        const points1 =
            hBracketPoints(span1Width, -1 * (this.bracketHeight - 1), 1);
        g1.querySelector('polyline')!.setAttribute('points', `${points1}`);

        // Set the width for the label; this will show ellipsis for the label
        // text if it is longer.
        // Leave a few pixels spacing if we can afford it, but don't go
        // shorter than the token width.
        const displayWidth = Math.max(span1Width, availableWidths[i] - 5);
        g.querySelector('foreignObject')!.setAttribute(
            'width', `${displayWidth}`);

        // If there's a second span, set up bracket
        // and draw arc from span1 -> span2 with the arrow on span1.
        if (edge.span2) {
          const g2 = g.querySelector('g.at-span2')!;
          // Set position within this row.
          g2.setAttribute(
              'transform', `translate(${this.getStartX(edge.span2)}, 0)`);
          // Compute span width in SVG units, based on rendered token width.
          const span2Width =
              this.getEndX(edge.span2) - this.getStartX(edge.span2);
          const points2 =
              hBracketPoints(span2Width, -1 * (this.bracketHeight - 1), 1);
          g2.querySelector('polyline')!.setAttribute('points', `${points2}`);

          // Draw arc.
          const startY =
              -1 * (this.bracketHeight + this.lineHeight + 1 /* pad */);
          const x1 =
              (this.getEndX(edge.span1) + this.getStartX(edge.span1)) / 2;
          let x2 = (this.getEndX(edge.span2) + this.getStartX(edge.span2)) / 2;
          // Adjust arc end to avoid overlapping arrows.
          // See //nlp/saft/rendering/sentence-html-renderer.js
          if (x2 > x1) {
            x2 -= (this.arcArrowSize + 2);
          } else {
            x2 += (this.arcArrowSize + 2);
          }
          // Adjust arc height based on edge length (# tokens between
          // midpoints). See nlp_saft::SentenceRenderer::CalculateDimensions()
          // from //nlp/saft/rendering/sentence-html-rendering.cc
          const mid1 = (edge.span1[1] + edge.span1[0]) / 2;
          const mid2 = (edge.span2[1] + edge.span2[0]) / 2;
          const l = Math.min(30, Math.abs(mid2 - mid1));
          const arcHeight = Math.min(
              this.arcBaseHeight + Math.round((10 - (l / 6.0)) * l),
              this.arcMaxHeight);
          g.querySelector('path.arc-path')!.setAttribute(
              'd', `${arcPath(startY, x1, x2, arcHeight, this.arcAspect)}`);
          g.querySelector('path.arc-arrow')!.setAttribute(
              'd', `${arcArrow(startY, x1, this.arcArrowSize)}`);
        }
      });

      // Set mouseover behavior for this layer.
      this.setMouseovers(layerGroup, orderedEdges);
    }

    // Set mouseover boxes to match the _visible_ size of the label container.
    this.shadowRoot!.querySelectorAll('g.edge-group').forEach(g => {
      matchBBox(
          g.querySelector('foreignObject') as SVGGElement,
          g.querySelector('rect.mousebox') as SVGRectElement);
    });

    // Stack layers vertically, using bounding boxes to avoid occlusion.
    this.positionLayers();
    // Finally, after everything is positioned, set the viewport for the whole
    // SVG.
    this.setSVGViewport();
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'span-graph-vis': SpanGraphVis;
  }
}
