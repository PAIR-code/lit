/**
 * @fileoverview Visualization for span-based annotations.
 *
 * This provides LIT modules for sparse span annotations using the
 * MultiSegmentAnnotations type. Annotations are rendered in-line as highlight
 * spans in running text, which is well-suited for tasks like QA or entity
 * recognition which have a small number of spans over a longer passage.
 *
 * Similar to span_graph_module, we provide two module classes:
 * - AnnotatedTextGoldModule for gold annotations (in the input data)
 * - AnnotatedTextModule for model predictions
 */

// tslint:disable:no-new-decorators
import '../elements/annotated_text_vis';

import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {AnnotationGroups, TextSegments} from '../elements/annotated_text_vis';
import {MultiSegmentAnnotations, TextSegment} from '../lib/lit_types';
import {IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, filterToKeys, findSpecKeys} from '../lib/utils';

import {styles as sharedStyles} from '../lib/shared_styles.css';

/** LIT module for model output. */
@customElement('annotated-text-gold-module')
export class AnnotatedTextGoldModule extends LitModule {
  static override title = 'Annotated Text (gold)';
  static override duplicateForExampleComparison = true;
  static override duplicateForModelComparison = false;
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => {
        return html`
      <annotated-text-gold-module model=${model} .shouldReact=${shouldReact}
        selectionServiceIndex=${selectionServiceIndex}>
      </annotated-text-gold-module>`;
      };

  static override get styles() {
    return sharedStyles;
  }

  override renderImpl() {
    const input = this.selectionService.primarySelectedInputData;
    if (!input) return null;

    const dataSpec = this.appState.currentDatasetSpec;

    // Text segment fields
    const segmentNames = findSpecKeys(dataSpec, TextSegment);
    const segments: TextSegments = filterToKeys(input.data, segmentNames);
    const segmentSpec = filterToKeys(dataSpec, segmentNames);

    // Annotation fields
    const annotationNames = findSpecKeys(dataSpec, MultiSegmentAnnotations);
    const annotations: AnnotationGroups =
        filterToKeys(input.data, annotationNames);
    const annotationSpec = filterToKeys(dataSpec, annotationNames);

    // If more than one model is selected, AnnotatedTextModule will be offset
    // vertically due to the model name header, while this one won't be.
    // So, add an offset so that the content still aligns when there is a
    // AnnotatedTextGoldModule and a AnnotatedTextModule side-by-side.
    const offsetForHeader = !this.appState.compareExamplesEnabled &&
        this.appState.currentModels.length > 1;

    // clang-format off
    return html`
      ${offsetForHeader? html`<div class='offset-for-module-header'></div>` : null}
      <annotated-text-vis .segments=${segments}
                          .segmentSpec=${segmentSpec}
                          .annotations=${annotations}
                          .annotationSpec=${annotationSpec}>
      </annotated-text-vis>`;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return findSpecKeys(datasetSpec, MultiSegmentAnnotations).length > 0;
  }
}

/** LIT module for model output. */
@customElement('annotated-text-module')
export class AnnotatedTextModule extends LitModule {
  static override title = 'Annotated Text (predicted)';
  static override duplicateForExampleComparison = true;
  static override duplicateForModelComparison = true;
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => {
        return html`
      <annotated-text-module model=${model} .shouldReact=${shouldReact}
        selectionServiceIndex=${selectionServiceIndex}>
      </annotated-text-module>`;
      };

  static override get styles() {
    return sharedStyles;
  }

  @observable private currentData?: IndexedInput;
  @observable private currentPreds: AnnotationGroups = {};

  override firstUpdated() {
    const getPrimarySelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getPrimarySelectedInputData, data => {
      this.updateToSelection(data);
    });
  }

  private async updateToSelection(input: IndexedInput|null) {
    if (input == null) {
      this.currentData = undefined;
      this.currentPreds = {};
      return;
    }
    // Before waiting for the backend call, update data and clear annotations.
    this.currentData = input;
    this.currentPreds = {};  // empty preds will render as (no data)

    const promise = this.apiService.getPreds(
        [input], this.model, this.appState.currentDataset,
        [MultiSegmentAnnotations], [], 'Retrieving annotations');
    const results = await this.loadLatest('answers', promise);
    if (results === null) return;

    // Update data again, in case selection changed rapidly.
    this.currentData = input;
    this.currentPreds = results[0] as AnnotationGroups;
  }

  override renderImpl() {
    if (!this.currentData) return null;

    const segmentNames =
        findSpecKeys(this.appState.currentDatasetSpec, TextSegment);
    const segments: TextSegments =
        filterToKeys(this.currentData.data, segmentNames);
    const segmentSpec =
        filterToKeys(this.appState.currentDatasetSpec, segmentNames);

    const outputSpec = this.appState.getModelSpec(this.model).output;
    const annotationSpec = filterToKeys(
        outputSpec, findSpecKeys(outputSpec, MultiSegmentAnnotations));
    // clang-format off
    return html`
      <annotated-text-vis .segments=${segments}
                          .segmentSpec=${segmentSpec}
                          .annotations=${this.currentPreds}
                          .annotationSpec=${annotationSpec}>
      </annotated-text-vis>`;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, MultiSegmentAnnotations);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'annotated-text-module': AnnotatedTextModule;
    'annotated-text-gold-module': AnnotatedTextGoldModule;
  }
}
