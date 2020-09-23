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
// taze: ResizeObserver from //third_party/javascript/typings/resize_observer_browser
import * as d3 from 'd3';
import {Dataset, Point3D, ScatterGL} from 'scatter-gl';
import {customElement, html, property} from 'lit-element';
import {TemplateResult} from 'lit-html';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {BatchRequestCache} from '../lib/caching';
import {IndexedInput, ModelsMap, Preds, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {ColorService} from '../services/services';

import {styles} from './embeddings_module.css';
import {styles as sharedStyles} from './shared_styles.css';

interface ProjectorOptions {
  displayName: string;
  // Name of backend interpreter.
  interpreterName: string;
}

/**
 * A LIT module showing a Scatter-GL rendering of the projected embeddings
 * for the input data.
 */
@customElement('embeddings-module')
export class EmbeddingsModule extends LitModule {
  static title = 'Embeddings';
  static template = () => {
    return html`<embeddings-module></embeddings-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  static duplicateForModelComparison = false;

  static projectorChoices: {[key: string]: ProjectorOptions} = {
    'pca': {displayName: 'PCA', interpreterName: 'pca'},
    'umap': {displayName: 'UMAP', interpreterName: 'umap'},
  };
  // Selection of one of the above configs.
  @observable private projectorName: string = 'umap';
  @computed
  get projector(): ProjectorOptions {
    return EmbeddingsModule.projectorChoices[this.projectorName];
  }

  // Actual projected points.
  @observable private projectedPoints: Point3D[] = [];

  private readonly colorService = app.getService(ColorService);
  private resizeObserver!: ResizeObserver;

  private scatterGL!: ScatterGL;

  /**
   * Cache for embeddings, so we don't need to retrieve the entire
   * set whenever new points are added.
   *
   * TODO(lit-dev): consider clearing these when dataset is changed, so we don't
   * use too much memory.
   */
  private readonly embeddingCache =
      new Map<string, BatchRequestCache<string, IndexedInput, Preds>>();

  @observable private selectedEmbeddingsIndex = 0;
  @observable private selectedLabelIndex = 0;

  @computed
  get currentInputIndicesById(): Map<string, number> {
    const indicesById = new Map<string, number>();
    this.appState.currentInputData.forEach((entry, index) => {
      indicesById.set(entry.id, index);
    });
    return indicesById;
  }

  @computed
  get embeddingOptions() {
    return this.appState.currentModels.flatMap((modelName: string) => {
      const modelSpec = this.appState.metadata.models[modelName].spec;
      const embKeys = findSpecKeys(modelSpec.output, 'Embeddings');
      return embKeys.map((fieldName) => ({modelName, fieldName}));
    });
  }

  /**
   * String versions of the above, for display and dropdown menu keys.
   */
  @computed
  get embeddingOptionNames() {
    return this.embeddingOptions.map(
        option => `${option.modelName}:${option.fieldName}`);
  }

  @computed
  private get displayLabels() {
    return this.appState.currentInputData.map((d: IndexedInput) => {
      const labelKey = Object.keys(d.data)[this.selectedLabelIndex];
      const label = d.data[labelKey];
      const added = d.meta['added'];
      return {label, added};
    });
  }

  @computed
  get scatterGLDataset(): Dataset|null {
    if (this.projectedPoints.length === 0) {
      return null;
    }
    const labels = this.displayLabels.slice(0, this.projectedPoints.length);
    return new Dataset(this.projectedPoints, labels);
  }

  private getEmbeddingCache(dataset: string, model: string) {
    const key = `${dataset}:${model}`;
    if (!this.embeddingCache.has(key)) {
      // Not found, create a new one.
      const keyFn = (d: IndexedInput) => d['id'];
      const requestFn = async (inputs: IndexedInput[]) => {
        return this.apiService.getPreds(
            inputs, model, dataset, ['Embeddings'], 'Fetching embeddings');
      };
      const cache = new BatchRequestCache(requestFn, keyFn);
      this.embeddingCache.set(key, cache);
    }
    return this.embeddingCache.get(key)!;
  }

  constructor() {
    super();
    // Default to PCA on large datasets.
    if (this.appState.currentInputData.length > 1000) {
      this.projectorName = 'pca';
    }
  }

  firstUpdated() {
    const scatterContainer =
        this.shadowRoot!.getElementById('scatter-gl-container')!;

    this.scatterGL = new ScatterGL(scatterContainer, {
      pointColorer: (i, selectedIndices, hoverIndex) =>
          this.pointColorer(i, selectedIndices, hoverIndex),
      onSelect: this.onSelect.bind(this),
      rotateOnStart: false
    });

    this.setupReactions();

    // Resize the scatter GL container.
    const container = this.shadowRoot!.getElementById('scatter-gl-container')!;
    this.resizeObserver = new ResizeObserver(() => {
      this.handleResize();
    });
    this.resizeObserver.observe(container);
  }

  private handleResize() {
    // Protect against resize when container isn't rendered, which can happen
    // during model switching.
    const scatterContainer =
        this.shadowRoot!.getElementById('scatter-gl-container')!;
    if (scatterContainer.offsetWidth > 0) {
      this.scatterGL.resize();
    }
  }

  /**
   * Trigger imperative updates, such as backend calls or scatterGL.
   */
  private setupReactions() {
    // Don't react immediately; we'll wait and make a single update.
    const getColorAll = () => this.colorService.all;
    this.react(getColorAll, selectedColorOption => {
      // pointColorer uses the latest settings from colorService automatically,
      // so to pick up the colors we just need to trigger a rerender on
      // scatterGL.
      this.updateScatterGL();
    });

    // Compute or update embeddings.
    // Since this is potentially expensive, set a small delay so mobx can batch
    // updates (e.g. if another component is adding several datapoints).
    // TODO(lit-dev): consider putting this delay somewhere shared,
    // like the LitModule class.
    const embeddingRecomputeData = () =>
        [this.appState.currentInputData, this.selectedEmbeddingsIndex,
         this.projectorName];
    this.reactImmediately(embeddingRecomputeData, () => {
      this.computeProjectedEmbeddings();
    }, {delay: 0.2});

    // Actually render the points.
    this.reactImmediately(() => this.scatterGLDataset, dataset => {
      this.updateScatterGL();
    });

    this.reactImmediately(
        () => this.selectionService.selectedIds, selectedIds => {
          const selectedIndices = this.uniqueIdsToIndices(selectedIds);
          this.scatterGL.select(selectedIndices);
        });
  }

  private updateScatterGL() {
    if (this.scatterGLDataset) {
      // Hack solution for a bug where scattergl doesn't render until it is
      // either interacted with or rotated. TODO(lit-team): update this once
      // b/160165921 is fixed.
      this.scatterGL.render(this.scatterGLDataset);
      this.scatterGL.render(this.scatterGLDataset);
    }
  }

  // Maps from unique identifiers of points in inputData to indices of points in
  // scatterGL.
  private uniqueIdsToIndices(ids: string[]): number[] {
    return ids.map(id => this.currentInputIndicesById.get(id))
               .filter(index => index !== undefined) as number[];
  }

  /**
   * Project embeddings using a server-side interpreter module.
   */
  private async computeBackendEmbeddings(interpreterName: string) {
    const embeddingsInfo = this.embeddingOptions[this.selectedEmbeddingsIndex];
    if (!embeddingsInfo) {
      return;
    }
    const {modelName, fieldName} = embeddingsInfo;

    const currentInputData = this.appState.currentInputData;
    const datasetName = this.appState.currentDataset;
    const projConfig = {
      'dataset_name': datasetName,
      'model_name': modelName,
      'field_name': fieldName,
      'proj_kw': {'n_components': 3},
    };
    // Projections will be returned for the whole dataset, including generated
    // examples, but the backend will ensure that the projection is trained on
    // only the original dataset. See components/projection.py.
    // TODO(lit-dev): add client-side cache so we don't need to re-fetch
    // embeddings for the entire dataset when a new datapoint is added.
    const promise = this.apiService.getInterpretations(
        currentInputData, modelName, datasetName, interpreterName, projConfig,
        'Fetching projections');
    const results =
        await this.loadLatest(`proj-${JSON.stringify(projConfig)}`, promise);
    if (results === null) return;
    this.projectedPoints = results.map((d: {'z': Point3D}) => d['z']);
  }

  private async computeProjectedEmbeddings() {
    // Clear projections if dataset is empty.
    if (!this.appState.currentInputData.length) {
      this.projectedPoints = [];
      return;
    }

    return this.computeBackendEmbeddings(this.projector.interpreterName);
  }

  /**
   * Color a point (different colors for user-added points vs points that were
   * part of the original dataset)
   * @param i index of the point to be colored.
   */
  private pointColorer(
      i: number, selectedIndices: Set<number>, hoveredIndex: number|null) {
    const currentPoint = this.appState.currentInputData[i];
    const color = this.colorService.getDatapointColor(currentPoint);

    // Add some transparency if not selected.
    const colorObject = d3.color(color)!;

    if (currentPoint != null &&
        !this.selectionService.isIdSelected(currentPoint.id)) {
      colorObject.opacity =
          this.selectionService.selectedInputData.length === 0 ? 0.7 : 0.1;
    }

    return colorObject.toString();
  }

  private onSelect(selectedIndices: number[]) {
    const ids = this.appState.currentInputData
                    .filter((data, index) => selectedIndices.includes(index))
                    .map((data) => data.id);
    this.selectionService.selectIds(ids);
  }

  render() {
    return html`
      <div class="container">
        <div class="toolbar-container flex-row">
          ${this.renderProjectorSelect()}
          ${this.renderEmbeddingsSelect()}
          ${this.renderLabelBySelect()}
        </div>
        <div id="scatter-gl-container"></div>
      </div>
    `;
  }

  renderSpinner() {
    return html`
      <div class="spinner-container">
        <lit-spinner size=${36} color="var(--app-dark-color)"></lit-spinner>
      </div>`;
  }

  renderProjectorSelect() {
    const options = EmbeddingsModule.projectorChoices;
    const htmlOptions = Object.keys(options).map((key) => {
      return html`
        <option value=${key} ?selected=${key === this.projectorName}>
          ${options[key].displayName}
        </option>
      `;
    });

    const handleChange = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      const selectedIndex = select?.selectedIndex || 0;
      this.projectorName = Object.keys(options)[selectedIndex];
    };

    return this.renderSelect(
        'Projector', htmlOptions, handleChange, this.projectorName);
  }

  renderEmbeddingsSelect() {
    const options = this.embeddingOptionNames;
    const htmlOptions = options.map((option, optionIndex) => {
      return html`
        <option value=${optionIndex}>${option}</option>
      `;
    });

    const handleChange = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      this.selectedEmbeddingsIndex = select?.selectedIndex || 0;
    };

    return this.renderSelect(
        'Embedding', htmlOptions, handleChange, options[0]);
  }

  renderLabelBySelect() {
    const inputData = this.appState.currentInputData;
    const noData = inputData === null || inputData.length === 0;
    const options: string[] = noData ? [] : Object.keys(inputData[0].data);

    const htmlOptions = options.map((option, optionIndex) => {
      return html`<option value=${optionIndex}>${option}</option>`;
    });

    const handleChange = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      this.selectedLabelIndex = select?.selectedIndex || 0;
    };

    return this.renderSelect(
        'Label by', htmlOptions, handleChange, options[0] ?? '');
  }

  renderSelect(
      label: string, options: TemplateResult[], onChange: (e: Event) => void,
      defaultValue: string) {
    return html`
      <div class="dropdown-container">
        <label class="dropdown-label">${label}</label>
        <select class="dropdown" @change=${onChange} .value=${defaultValue}>
          ${options}
        </select>
      </div>
    `;
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'Embeddings');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'embeddings-module': EmbeddingsModule;
  }
}
