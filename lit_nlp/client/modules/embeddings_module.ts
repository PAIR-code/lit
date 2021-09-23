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
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {TemplateResult} from 'lit';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {BatchRequestCache} from '../lib/caching';
import {CallConfig, IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
import {ColorService, FocusService} from '../services/services';

import {styles} from './embeddings_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

interface ProjectorOptions {
  displayName: string;
  // Name of backend interpreter.
  interpreterName: string;
}

interface ProjectionBackendResult {
  z: Point3D;
}

interface NearestNeighborsResult {
  'id': string;
}

const NN_INTERPRETER_NAME = 'nearest neighbors';
// TODO(lit-dev): Make the number of nearest neighbors configurable in the UI.
const DEFAULT_NUM_NEAREST = 10;

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

  static override get styles() {
    return [sharedStyles, styles];
  }

  static override duplicateForModelComparison = false;

  static projectorChoices: {[key: string]: ProjectorOptions} = {
    'pca': {displayName: 'PCA', interpreterName: 'pca'},
    'umap': {displayName: 'UMAP', interpreterName: 'umap'},
  };

  static override numCols = 3;

  // Selection of one of the above configs.
  @observable private projectorName: string = 'umap';
  @computed
  get projector(): ProjectorOptions {
    return EmbeddingsModule.projectorChoices[this.projectorName];
  }

  // Actual projected points.
  @observable private projectedPoints: Point3D[] = [];

  private readonly colorService = app.getService(ColorService);
  private readonly focusService = app.getService(FocusService);
  private resizeObserver!: ResizeObserver;

  private scatterGL!: ScatterGL;

  /**
   * Cache for embeddings, so we don't need to retrieve the entire
   * set whenever new points are added.
   *
   * TODO(lit-dev): consider clearing these when dataset is changed, so we don't
   * use too much memory.
   */
  private readonly embeddingCache = new Map<
      string,
      BatchRequestCache<string, IndexedInput, ProjectionBackendResult>>();

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
      const added = d.meta['added'] ? 1 : 0;
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

  /**
   * Return a frontend embedding cache, so we don't need to re-fetch the entire
   * dataset when new points are added.
   */
  private getEmbeddingCache(
      dataset: string, model: string, projector: string, config: CallConfig) {
    const key = `${dataset}:${model}:${projector}:${JSON.stringify(config)}`;
    if (!this.embeddingCache.has(key)) {
      // Not found, create a new one.
      const keyFn = (d: IndexedInput) => d['id'];
      const requestFn =
          async(inputs: IndexedInput[]): Promise<ProjectionBackendResult[]> => {
        return this.apiService.getInterpretations(
            inputs, model, dataset, projector, config, 'Fetching projections');
      };
      const cache = new BatchRequestCache(requestFn, keyFn);
      this.embeddingCache.set(key, cache);
    }
    return this.embeddingCache.get(key)!;
  }

  private async getNearestNeighbors(
      example: IndexedInput, numNeighbors: number = DEFAULT_NUM_NEAREST) {
    const {modelName, fieldName} =
        this.embeddingOptions[this.selectedEmbeddingsIndex];
    const datasetName = this.appState.currentDataset;
    const config: CallConfig = {
      'dataset_name': datasetName,
      'embedding_name': fieldName,
      'num_neighbors': numNeighbors,
    };

    // All indexed inputs in the dataset are passed in, with the main example
    // id (to get nearest neighbors for) specified in the config.
    // TODO(b/178210779): Enable caching in the component's predict call.
    const result = await this.apiService.getInterpretations(
        [example], modelName, this.appState.currentDataset, NN_INTERPRETER_NAME,
        config, `Running ${NN_INTERPRETER_NAME}`);

    if (result === null) return;

    const nearestIds = result[0]['nearest_neighbors'].map(
        (neighbor: NearestNeighborsResult) => {
          return neighbor['id'];
        });

    this.selectionService.selectIds(nearestIds);
  }

  constructor() {
    super();
    // Default to PCA on large datasets.
    if (this.appState.currentInputData.length > 1000) {
      this.projectorName = 'pca';
    }
  }

  override firstUpdated() {
    const scatterContainer =
        this.shadowRoot!.getElementById('scatter-gl-container')!;

    this.scatterGL = new ScatterGL(scatterContainer, {
      pointColorer: (i, selectedIndices, hoverIndex) =>
          this.pointColorer(i, selectedIndices, hoverIndex),
      onSelect: this.onSelect.bind(this),
      onHover: this.onHover.bind(this),
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
    this.react(getColorAll, allColorOptions => {
      // pointColorer uses the latest settings from colorService automatically,
      // so to pick up the colors we just need to trigger a rerender on
      // scatterGL.
      this.updateScatterGL();
    });
    this.react(() => this.focusService.focusData, focusData => {
      this.updateScatterGL();
    });
    this.react(() => this.selectionService.primarySelectedId, primaryId => {
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

  private async computeProjectedEmbeddings() {
    // Clear projections if dataset is empty.
    if (!this.appState.currentInputData.length) {
      this.projectedPoints = [];
      return;
    }

    const embeddingsInfo = this.embeddingOptions[this.selectedEmbeddingsIndex];
    if (!embeddingsInfo) {
      return;
    }
    const {modelName, fieldName} = embeddingsInfo;

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
    const embeddingRequestCache = this.getEmbeddingCache(
        datasetName, modelName, this.projector.interpreterName, projConfig);
    const promise = embeddingRequestCache.call(this.appState.currentInputData);
    const results =
        await this.loadLatest(`proj-${JSON.stringify(projConfig)}`, promise);
    if (results === null) return;
    this.projectedPoints = results.map((d: {'z': Point3D}) => d['z']);
  }

  /**
   * Color a point (different colors for user-added points vs points that were
   * part of the original dataset)
   * @param i index of the point to be colored.
   */
  private pointColorer(
      i: number, selectedIndices: Set<number>, hoveredIndex: number|null) {
    const currentPoint = this.appState.currentInputData[i];
    let color = this.colorService.getDatapointColor(currentPoint);

    const isSelected = currentPoint != null &&
        this.selectionService.isIdSelected(currentPoint.id);
    const isPrimarySelected = currentPoint != null &&
        this.selectionService.primarySelectedId === currentPoint.id;
    const isHovered = this.focusService.focusData != null &&
        this.focusService.focusData.datapointId === currentPoint.id &&
        this.focusService.focusData.io == null;

    if (isHovered) {
      color = 'red';
    }
    // Add some transparency if not selected or hovered.
    const colorObject = d3.color(color)!;

    if (!isSelected && !isHovered) {
      colorObject.opacity =
          this.selectionService.selectedInputData.length === 0 ? 0.7 : 0.1;
    } else if (isSelected && !isPrimarySelected) {
      colorObject.opacity = .5;
    }

    return colorObject.toString();
  }

  private onSelect(selectedIndices: number[]) {
    const ids = this.appState.currentInputData
                    .filter((data, index) => selectedIndices.includes(index))
                    .map((data) => data.id);
    this.selectionService.selectIds(ids);
  }

  private onHover(hoveredIndex: number|null) {
    if (hoveredIndex == null) {
      this.focusService.clearFocus();
    } else {
      this.focusService.setFocusedDatapoint(
          this.appState.currentInputData[hoveredIndex].id);
    }
  }

  override render() {
    const onSelectNearest = () => {
      if (this.selectionService.primarySelectedInputData != null) {
        this.getNearestNeighbors(
            this.selectionService.primarySelectedInputData);
      }
    };
    const disabled = this.selectionService.selectedIds.length !== 1;
    return html`
      <div class="container">
        <div class="toolbar-container flex-row">
          ${this.renderProjectorSelect()}
          ${this.renderEmbeddingsSelect()}
          ${this.renderLabelBySelect()}
        </div>
        <div class="toolbar-container flex-row" id="select-button-container">
          <button class="hairline-button selected-nearest-button" ?disabled=${
        disabled}
        @click=${onSelectNearest} title=${
        disabled ? 'Select a single point to use this feature' : ''}
        >Select ${DEFAULT_NUM_NEAREST} nearest neighbors</button>
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

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'Embeddings');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'embeddings-module': EmbeddingsModule;
  }
}
