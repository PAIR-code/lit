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
 * Module within LIT for showing sequence and span tagging
 * results.
 */

// tslint:disable:no-new-decorators
import '../elements/span_graph_vis';
import '../elements/span_graph_vis_vertical';

import {customElement} from 'lit/decorators';
import {css, html} from 'lit';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {AnnotationLayer, SpanGraph} from '../elements/span_graph_vis_vertical';
import {EdgeLabel, SpanLabel} from '../lib/dtypes';
import {EdgeLabels, SequenceTags, SpanLabels, LitTypeTypesList, LitTypeWithAlign, TextSegment, Tokens} from '../lib/lit_types';
import {IndexedInput, Input, ModelInfoMap, Preds, Spec} from '../lib/types';
import {findSpecKeys} from '../lib/utils';

import {styles as sharedStyles} from '../lib/shared_styles.css';

interface FieldNameMultimap {
  [fieldName: string]: string[];
}

interface Annotations {
  [tokenKey: string]: SpanGraph;
}

// Shared by gold and preds modules.
const moduleStyles = css`
  .outer-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
  }

  .token-group {
    padding-top: 30pt;
  }

  .field-title {
    padding: 4px;
  }
`;

const supportedPredTypes: LitTypeTypesList =
    [SequenceTags, SpanLabels, EdgeLabels];

/**
 * Convert sequence tags to a list of length-1 span labels.
 */
function tagsToEdges(tags: string[]): EdgeLabel[] {
  return tags.map((label: string, i: number) => {
    return {span1: [i, i + 1], label} as EdgeLabel;
  });
}

/**
 * Convert span labels to single-sided edge labels.
 */
function spansToEdges(spans: SpanLabel[]): EdgeLabel[] {
  return spans.map(
      d => ({span1: [d.start, d.end], label: d.label as string} as EdgeLabel));
}

function mapTokenToTags(spec: Spec): FieldNameMultimap {
  const tagKeys = findSpecKeys(spec, supportedPredTypes);
  const tokenKeys = findSpecKeys(spec, Tokens);

  // Make a mapping of token keys to one or more tag sets
  const tokenToTags: FieldNameMultimap = {};
  for (const tagKey of tagKeys) {
    const {align: tokenKey} = spec[tagKey] as LitTypeWithAlign;
    if (!tokenKeys.includes(tokenKey)) {
      continue;
    }
    if (tokenToTags[tokenKey] == null) {
      tokenToTags[tokenKey] = [];
    }
    tokenToTags[tokenKey].push(tagKey);
  }
  return tokenToTags;
}

function parseInput(data: Input|Preds, spec: Spec): Annotations {
  const tokenToTags = mapTokenToTags(spec);

  // Render a row for each set of tokens
  const ret: Annotations = {};
  for (const tokenKey of Object.keys(tokenToTags)) {
    const annotationLayers: AnnotationLayer[] = [];
    for (const tagKey of tokenToTags[tokenKey]) {
      let edges = data[tagKey];
      let hideBracket = false;
      // Temporary workaround: if we manually create a new datapoint, the span
      // or tag field may be "" rather than [].
      // TODO(lit-team): remove this once the datapoint editor is type-safe
      // for structured fields.
      if (edges.length === 0) {
        edges = [];
      }
      if (spec[tagKey] instanceof SequenceTags) {
        edges = tagsToEdges(edges);
        hideBracket = true;
      } else if (spec[tagKey] instanceof SpanLabels) {
        edges = spansToEdges(edges);
      }
      annotationLayers.push({name: tagKey, edges, hideBracket});
    }
    // Try to infer tokens from text, if that field is empty.
    let tokens = data[tokenKey];
    if (tokens.length === 0) {
      const textKey = findSpecKeys(spec, TextSegment)[0];
      tokens = data[textKey].split();
    }
    ret[tokenKey] = {tokens, layers: annotationLayers};
  }
  return ret;
}

function renderTokenGroups(
    data: Annotations, spec: Spec, orientation: 'horizontal'|'vertical') {
  const tokenToTags = mapTokenToTags(spec);
  const visElement = (data: SpanGraph, showLayerLabel: boolean) => {
    if (orientation === 'vertical') {
      return html`<span-graph-vis-vertical .data=${data} .showLayerLabel=${
          showLayerLabel}></span-graph-vis-vertical>`;
    } else {
      return html`<span-graph-vis .data=${data} .showLayerLabel=${
          showLayerLabel}></span-graph-vis>`;
    }
  };
  // clang-format off
  return html`${Object.keys(tokenToTags).map(tokenKey => {
    const labelHere = data[tokenKey]?.layers?.length === 1;
    return html`
      <div id=${tokenKey} class="token-group">
        ${labelHere ?
          html`<div class='field-title'>${data[tokenKey].layers[0].name}</div>`
          : null}
        ${visElement(data[tokenKey], !labelHere)}
      </div>
    `;
  })}`;
  // clang-format on
}

/** Gold predictions module class. */
@customElement('span-graph-gold-module')
export class SpanGraphGoldModule extends LitModule {
  static override title = 'Structured Prediction (gold)';
  static override duplicateForExampleComparison = true;
  static override duplicateForModelComparison = false;
  static override duplicateAsRow = false;
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <span-graph-gold-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </span-graph-gold-module>`;
  static orientation = 'horizontal';

  @computed
  get dataSpec() {
    return this.appState.currentDatasetSpec;
  }

  @computed
  get goldDisplayData(): Annotations {
    const input = this.selectionService.primarySelectedInputData;
    if (input === null) {
      return {};
    } else {
      return parseInput(input.data, this.dataSpec);
    }
  }

  static override get styles() {
    return [sharedStyles, moduleStyles];
  }

  // tslint:disable:no-any
  override renderImpl() {
    // If more than one model is selected, SpanGraphModule will be offset
    // vertically due to the model name header, while this one won't be.
    // So, add an offset so that the content still aligns when there is a
    // SpanGraphGoldModule and a SpanGraphModule side-by-side.
    const offsetForHeader = !this.appState.compareExamplesEnabled &&
        this.appState.currentModels.length > 1;
    // clang-format off
    return html`
      ${offsetForHeader? html`<div class='offset-for-module-header'></div>` : null}
      <div id="gold-group" class='outer-container'>
        ${
        renderTokenGroups(
            this.goldDisplayData, this.dataSpec,
            (this.constructor as any).orientation)}
      </div>
    `;
    // clang-format on
  }
  // tslint:enable:no-any

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    const hasTokens = findSpecKeys(datasetSpec, Tokens).length > 0;
    const hasSupportedPreds =
        findSpecKeys(datasetSpec, supportedPredTypes).length > 0;
    return (hasTokens && hasSupportedPreds);
  }
}

/** Model output module class. */
@customElement('span-graph-module')
export class SpanGraphModule extends LitModule {
  static override title = 'Structured Prediction (model preds)';
  static override duplicateForExampleComparison = true;
  static override duplicateAsRow = false;
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <span-graph-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </span-graph-module>`;
  static orientation = 'horizontal';

  @computed
  get predSpec() {
    return this.appState.getModelSpec(this.model).output;
  }

  // This is updated with an API call, via a reaction.
  @observable predDisplayData: Annotations = {};

  private async updatePredDisplayData(input: IndexedInput|null) {
    if (input === null) {
      this.predDisplayData = {};
    } else {
      const promise = this.apiService.getPreds(
          [input], this.model, this.appState.currentDataset,
          [Tokens, ...supportedPredTypes]);

      const results = await this.loadLatest('getPreds', promise);
      if (!results) return;

      this.predDisplayData = parseInput(results[0], this.predSpec);
    }
  }

  static override get styles() {
    return [sharedStyles, moduleStyles];
  }

  override firstUpdated() {
    this.reactImmediately(
        () => this.selectionService.primarySelectedInputData, input => {
          this.updatePredDisplayData(input);
        });
  }

  // tslint:disable:no-any
  override renderImpl() {
    return html`
      <div id="pred-group" class='outer-container'>
        ${
        renderTokenGroups(
            this.predDisplayData, this.predSpec,
            (this.constructor as any).orientation)}
      </div>
    `;
  }
  // tslint:enable:no-any

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    const models = Object.keys(modelSpecs);
    for (let modelNum = 0; modelNum < models.length; modelNum++) {
      const spec = modelSpecs[models[modelNum]].spec;
      const hasTokens = findSpecKeys(spec.output, Tokens).length > 0;
      const hasSupportedPreds =
          findSpecKeys(spec.output, supportedPredTypes).length > 0;
      if (hasTokens && hasSupportedPreds) {
        return true;
      }
    }
    return false;
  }
}

// tslint:disable:class-as-namespace

/** Gold predictions module class. */
@customElement('span-graph-gold-module-vertical')
export class SpanGraphGoldModuleVertical extends SpanGraphGoldModule {
  static override duplicateAsRow = true;
  static override orientation = 'vertical';
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <span-graph-gold-module-vertical model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </span-graph-gold-module-vertical>`;
}

/** Model output module class. */
@customElement('span-graph-module-vertical')
export class SpanGraphModuleVertical extends SpanGraphModule {
  static override duplicateAsRow = true;
  static override orientation = 'vertical';
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <span-graph-module-vertical model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </span-graph-module-vertical>`;
}

// tslint:enable:class-as-namespace

declare global {
  interface HTMLElementTagNameMap {
    'span-graph-gold-module': SpanGraphGoldModule;
    'span-graph-module': SpanGraphModule;
    // TODO(b/172979677): make these parameterized versions, rather than
    // separate classes.
    'span-graph-gold-module-vertical': SpanGraphGoldModuleVertical;
    'span-graph-module-vertical': SpanGraphModuleVertical;
  }
}
