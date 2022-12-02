/**
 * @fileoverview Visualization for multiple segment annotations.
 */

// tslint:disable:no-new-decorators
import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import { html, LitElement} from 'lit';
import {TemplateResult} from 'lit';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';

import {getVizColor} from '../lib/colors';
import {ReactiveElement} from '../lib/elements';
import {SpanLabel} from '../lib/dtypes';
import {URLLitType} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatSpanLabel} from '../lib/types';

import {styles} from './annotated_text_vis.css';


/** LIT spec info for segments */
export interface SegmentSpec {
  // tslint:disable-next-line:enforce-name-casing
  [name: string]: {__name__: string};
}

/** Input data */
export interface TextSegments {
  [name: string]: string;
}

/** Individual annotations, which contain one or more spans. */
export interface AnnotationCluster {
  label: string;
  spans: SpanLabel[];
  score?: number;
}

/** Annotations (from model) */
export interface AnnotationGroups {
  [group: string]: AnnotationCluster[];
}

/** LIT spec info for annotations */
export interface AnnotationSpec {
  [group: string]: {exclusive: boolean, background: boolean};
}

/* For coloring spans */
interface StyledSpan extends SpanLabel {
  color?: string;
}

// Construct a striped background using linear-gradient
function makeStripedBackground(angle: string, colors: string[]) {
  return colors.map((c, i) => {
    const end = 100*(i+1)/colors.length;
    return `linear-gradient(${angle}, ${c} ${end}%, rgba(0,0,0,0) ${end}%)`;
  }).join(", ");
}

const DEFAULT_SPAN_BACKGROUND = '#e4f7fb';

/** Annotated text passage. */
@customElement('annotated-text')
class AnnotatedText extends LitElement {
  /* Data binding */
  @property({type: String}) text: string = "";
  @property({type: Array}) spans: StyledSpan[] = [];  // aligned to this segment
  @property({type: Boolean}) isURL: boolean = false;  // if true, make a link

  static override get styles() {
    return [sharedStyles, styles];
  }

  override render() {
    // Gather all endpoints, for chunking.
    const allEndpoints = new Set<number>();
    for (const span of this.spans) {
      allEndpoints.add(span.start);
      allEndpoints.add(span.end);
    }
    let endpoints = [0, ...allEndpoints, this.text.length];
    endpoints = endpoints.sort((a,b) => a-b);
    // Create a multimap of chunk index -> [span indices]
    const chunkToSpanIndices: number[][] = endpoints.map(() => []);
    for (let j = 0; j < this.spans.length; j++) {
      // Remap to chunk indices.
      const start = endpoints.indexOf(this.spans[j].start);
      const end = endpoints.indexOf(this.spans[j].end);
      for (let i = start; i < end; i++) {
        chunkToSpanIndices[i].push(j);
      }
    }

    // Render text as span chunks.
    const chunks: TemplateResult[] = [];
    for (let i = 0; i < endpoints.length - 1; i++) {
      const chunk = this.text.slice(endpoints[i], endpoints[i+1]);
      const spans = chunkToSpanIndices[i].map(j => this.spans[j]);
      const classes = classMap({'span-active': spans.length > 0});
      const style: {[name: string]: string} = {};
      if (spans.length > 1) {
        style['background'] = makeStripedBackground(
            '0deg', spans.map(s => s.color ?? DEFAULT_SPAN_BACKGROUND));
      } else if (spans.length > 0) {
        style['background'] = spans[0]?.color ?? DEFAULT_SPAN_BACKGROUND;
      }
      chunks.push(html`<span class=${classes} style=${styleMap(style)}>${chunk}</span>`);
    }
    if (this.isURL) {
      return html`<a href=${this.text} target="_blank">${chunks}</a>`;
    } else {
      return html`${chunks}`;
    }
  }
}

/** Multi-segment annotation visualization class. */
@customElement('annotated-text-vis')
export class AnnotatedTextVis extends ReactiveElement {
  /* Data binding */
  @observable @property({type: Object}) segments: TextSegments = {};
  @observable @property({type: Object}) segmentSpec: SegmentSpec = {};
  @observable @property({type: Object}) annotations: AnnotationGroups = {};
  /* Spec info for each annotation field, such as exclusivity bit. */
  @observable @property({type: Object}) annotationSpec: AnnotationSpec = {};

  /* groupName -> set of indices, parallel to this.annotations */
  @observable annotationVisibility: {[groupName: string]: Set<number>} = {};

  @computed get groupColors() {
    // Colors for each annotation group.
    // TODO(lit-dev): different colors for each cluster if non-exclusive?
    const groupColors: {[groupName: string]: string} = {};
    const specKeys = Object.keys(this.annotationSpec);
    for (let i = 0; i < specKeys.length; i++) {
      const name = specKeys[i];
      groupColors[name] = getVizColor('pastel', i).color;
    }
    return groupColors;
  }

  @computed get activeAnnotations() {
    // Group annotation spans by text segment.
    const activeAnnotations: {[segment: string]: SpanLabel[]} = {};
    for (const segmentName of Object.keys(this.segments)) {
      activeAnnotations[segmentName] = [];
    }
    for (const groupName of Object.keys(this.annotations)) {
      this.annotations[groupName].forEach((cluster, i) => {
        if (!this.annotationVisibility[groupName].has(i)) return;
        for (const span of cluster.spans) {
          // Identify each new span, labeled by cluster and span label,
          // and aligned to the annotation group.
          const newSpan = Object.assign({}, span,
            {label: `${cluster.label}${span.label ? ':' : ''}${span.label}`,
             align: groupName, color: this.groupColors[groupName]});
          activeAnnotations[span.align!].push(newSpan);
        }
      });
    }
    return activeAnnotations;
  }

  static override get styles() {
    return [sharedStyles, styles];
  }

  override connectedCallback() {
    super.connectedCallback();
    this.reactImmediately(() => this.annotations, annotations => {
      this.annotationVisibility = {};
      for (const groupName of Object.keys(annotations)) {
        this.annotationVisibility[groupName] = new Set<number>();
        // Set default visibility.
        if (this.annotationSpec[groupName]?.background) {
          // Nothing visible by default in this case.
        } else if (this.annotationSpec[groupName]?.exclusive) {
          // For 'exclusive' sets, show the first cluster only.
          this.annotationVisibility[groupName].add(0);
        } else {
          // For normal sets, show all clusters.
          for (let i = 0; i < annotations[groupName].length; i++) {
            this.annotationVisibility[groupName].add(i);
          }
        }
      }
    });
  }

  /* Render a text segment, with active annotations. */
  renderTextSegment(name: string) {
    const text = this.segments[name];
    const spans = this.activeAnnotations[name];
    const isURL = this.segmentSpec[name] instanceof URLLitType;
    return html`
      <div class='group'>
        <div class='group-title'>${name}</div>
        <annotated-text .text=${text} .spans=${spans} ?isURL=${isURL}>
        </annotated-text>
      </div>
    `;
  }

  /* Render controls for a single annotation cluster (such as mention). */
  renderCluster(groupName: string, i: number) {
    const cluster = this.annotations[groupName][i];
    const highlight = this.annotationVisibility[groupName].has(i);
    const onClickCluster = () => {
      if (this.annotationVisibility[groupName].has(i)) {
        // De-select if currently active.
        this.annotationVisibility[groupName].delete(i);
      } else {
        if (this.annotationSpec[groupName].exclusive) {
          // If exclusive, clear all others.
          this.annotationVisibility[groupName].clear();
        }
        // Select if not active.
        this.annotationVisibility[groupName].add(i);
      }
    };
    // clang-format off
    return html`
      <div class=${classMap({'cluster-container': true, 'cluster-active': highlight})}
           @click=${onClickCluster}>
        <div class='cluster-label'>${cluster.label}</div>
        <div class='cluster-spans'>
          ${cluster.spans.map(
            span => html`<div class='span'>${formatSpanLabel(span)}</div>`)}
        </div>
      </div>
    `;
    // clang-format on
  }

  /* Render a group of annotations (a single MultiSegmentAnnotations field). */
  renderAnnotationGroup(groupName: string) {
    const color = this.groupColors[groupName];
    const annotations = this.annotations?.[groupName];
    // clang-format off
    return html`
      <div class='group' style=${styleMap({"--group-color": color})}>
        <div class='group-title'>${groupName}</div>
        ${annotations ?
          annotations.map((a, i) => this.renderCluster(groupName, i)) :
          html`<span class='no-data'>(no data)</span>`}
      </div>
    `;
    // clang-format on
  }

  override render() {
    const renderedSegments = Object.keys(this.segmentSpec)
        .map(name => this.renderTextSegment(name));

    const renderedAnnotations = Object.keys(this.annotationSpec)
        .map(name => this.renderAnnotationGroup(name));

    // clang-format off
    return html`
      <div class='main-container'>
        <div class='segments-column'>${renderedSegments}</div>
        <div class='annotations-column'>${renderedAnnotations}</div>
      </div>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'annotated-text-vis': AnnotatedTextVis;
    'annotated-text': AnnotatedText;
  }
}
