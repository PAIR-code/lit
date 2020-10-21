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
import {action, computed, observable, toJS} from 'mobx';
import {v4 as uuidv4} from 'uuid';

import {IndexedInput, Input, LitMetadata, ModelsMap, ModelSpec, Spec} from '../lib/types';

import {ApiService} from './api_service';
import {LitService} from './lit_service';
import {StatusService} from './services';
import {StateObservedByUrlService, UrlConfiguration} from './url_service';


type Id = string;
type ModelName = string;
type DatasetName = string;
type IndexedInputMap = Map<Id, IndexedInput>;

/**
 * App state singleton, responsible for coordinating shared state between
 * different LIT modules and providing access to that state through
 * observable properties. AppState is also responsible for initialization
 * (although this may want to be factored out into a complementary class).
 */
export class AppState extends LitService implements StateObservedByUrlService {
  constructor(
      private readonly apiService: ApiService,
      private readonly statusService: StatusService) {
    super();
  }

  /** Set by urlService.syncStateToUrl */
  private urlConfiguration!: UrlConfiguration;

  @observable initialized = false;

  @observable metadata!: LitMetadata;
  @observable currentModels: string[] = [];
  @observable compareExamplesEnabled: boolean = false;
  @observable layoutName!: string;

  /**
   * Enforce setting currentDataset through the setCurrentDataset method by
   * making the currentDatasetInternal private...
   */
  @observable private currentDatasetInternal: string = '';
  @computed
  get currentDataset(): string {
    return this.currentDatasetInternal;
  }

  /**
   * When we set the current dataset, we need to load new data
   * from the server if the data is not already loaded...
   */
  @action
  setCurrentDataset(dataset: string, shouldLoadDataset = false) {
    this.currentDatasetInternal = dataset;
    if (!this.currentInputDataIsLoaded && shouldLoadDataset) {
      // TODO (b/154508392): We need to do this in an async/race condition safe
      // way and guarantee that we won't fetch multiple datasets simultaneously
      this.loadData();
    }
  }

  @computed
  get currentDatasetSpec(): Spec {
    return this.metadata.datasets[this.currentDataset].spec;
  }

  @observable
  private readonly inputData = new Map<DatasetName, IndexedInputMap>();

  private makeEmptyInputs(): IndexedInputMap {
    return new Map<Id, IndexedInput>();
  }

  @computed
  get currentInputDataById(): IndexedInputMap {
    if (!this.currentDataset) return this.makeEmptyInputs();
    const data = this.inputData.get(this.currentDataset);
    return data ? data : this.makeEmptyInputs();
  }

  @computed
  get currentInputData(): IndexedInput[] {
    return [...this.currentInputDataById.values()];
  }

  @computed
  get currentInputDataIsLoaded(): boolean {
    return this.currentInputData.length > 0;
  }

  /** Returns all data keys related to required model inputs. */
  @computed
  get currentModelRequiredInputSpecKeys(): string[] {
    // Add all required keys from current model input specs.
    const keys = new Set<string>();
    Object.values(this.currentModelSpecs).forEach((modelSpec: ModelSpec) => {
      Object.keys(modelSpec.spec.input).forEach(key => {
        if (modelSpec.spec.input[key].required === true) {
          keys.add(key);
        }
      });
    });
    return [...keys];
  }

  @computed
  get currentInputDataKeys(): string[] {
    return Object.keys(this.currentDatasetSpec);
  }

  @computed
  get indicesById(): Map<string, number> {
    const idToIndex = new Map<string, number>();
    this.currentInputData.forEach((entry, index) => {
      idToIndex.set(entry.id, index);
    });
    return idToIndex;
  }

  // Filter to only generated data points
  @computed
  get generatedDataPoints() {
    return this.currentInputData.filter((d: IndexedInput) => d.meta.added);
  }

  /**
   * Find the numerical index of the given id.
   * Returns -1 if id is null or not found.
   */
  getIndexById(id: string|null) {
    if (id == null) return -1;
    const index = this.indicesById.get(id);
    return index ?? -1;
  }

  getCurrentInputDataById(id: string): IndexedInput|null {
    const entry = this.currentInputDataById.get(id);
    return entry ? entry : null;
  }

  getExamplesById(ids: string[]): IndexedInput[] {
    const inputs: IndexedInput[] = [];
    ids.forEach(id => {
      const input = this.currentInputDataById.get(id);
      if (!input) {
        console.error(`input key ${
            input} was not found in the currently loaded dataset.`);
      } else {
        inputs.push(input);
      }
    });
    return inputs;
  }

  /**
   * Return the ancestry [id, parentId, grandParentId, ...] of an id,
   * by recursively following parent pointers.
   */
  getAncestry(id: string): string[] {
    const ret: string[] = [];
    while (id) {
      ret.push(id);
      id = this.getCurrentInputDataById(id)?.meta['parentId'];
    }
    return ret;
  }

  /**
   * Select models.
   */
  @action
  async setCurrentModels(currentModels: string[]) {
    this.currentModels = currentModels;
  }

  /**
   * Get the configs of only the current models.
   */
  @computed
  get currentModelSpecs() {
    const allModelSpecs = this.metadata.models;

    // Get the specs of only the selected models.
    const currentModelSpecs: ModelsMap = {};
    Object.keys(allModelSpecs).forEach(modelName => {
      if (this.currentModels.includes(modelName)) {
        currentModelSpecs[modelName] = allModelSpecs[modelName];
      }
    });
    return currentModelSpecs;
  }

  /**
   * Get the input and output spec for a particular model.
   */
  getModelSpec(modelName: string): {
    input: Spec,
    output: Spec,
  } {
    return this.metadata.models[modelName].spec;
  }

  //=================================== Generation logic
  /**
   * Create and add new datapoints from the output of a generator.
   * @param data list-of-lists of inputs. Inner lists are generations derived
   * from a single input.
   * @param parentIds list of ids of parents, corresponding to the outer index
   * of data.
   * @param source name of the generator that created these points.
   */
  async createNewDatapoints(
      data: Input[][], parentIds: string[],
      source: string, rule?: string): Promise<IndexedInput[]> {
    // Tag all datapoints created as a group.
    const creationId = uuidv4();

    let datapoints: IndexedInput[] = [];
    // Loop through new counterfactuals. Outer loop for input examples,
    // inner loop for list of counterfactuals for each input.
    for (let i = 0; i < data.length; i++) {
      for (let j = 0; j < data[i].length; j++) {
        const point: IndexedInput = {
          'data': data[i][j],
          'id': '',
          'meta': {
            'parentId': parentIds[i],
            'source': source,
            'rule': rule,
            'creationId': creationId,
            'added': 1,
            'isFavorited': false
          }
        };

        datapoints.push(point);
      }
    }

    // Fill in unique IDs for the new datapoints.
    datapoints = await this.apiService.getDatapointIds(datapoints);

    // Update input data for new datapoints.
    this.commitNewDatapoints(datapoints);
    return datapoints;
  }


  /**
   * Atomically commit a list of new datapoints.
   */
  @action
  private commitNewDatapoints(datapoints: IndexedInput[]) {
    datapoints.forEach(entry => {
      // If the new datapoint already exists in the input data, do not overwrite
      // it with this new copy, as that will cause issues with datapoint parent
      // tracking (an infinite loop of parent pointers).
      if (this.currentInputDataById.has(entry.id)) {
        console.log(
            'Attempted to add existing datapoint, ignoring add request.',
            entry);
      } else {
        this.currentInputDataById.set(entry.id, entry);
      }
    });
  }


  //=================================== Initialization logic
  @action
  async initialize() {
    const {urlConfiguration} = this;
    console.log('[LIT - url configuration]', urlConfiguration);

    const info = await this.apiService.getInfo();
    console.log('[LIT - metadata]', toJS(info));
    this.metadata = info;
    this.layoutName = urlConfiguration.layout || this.metadata.defaultLayout;

    this.currentModels = this.determineCurrentModelsFromUrl(urlConfiguration);
    this.setCurrentDataset(
        this.determineCurrentDatasetFromUrl(urlConfiguration),
        /** should Load Data */ false);

    await this.loadData();

    this.initialized = true;
  }

  async loadData() {
    if (this.currentDataset) {
      const inputResponse =
          await this.apiService.getInputs(this.currentDataset);
      this.updateInputData(this.currentDataset, inputResponse);
    }
  }

  private updateInputData(dataset: string, data: IndexedInput[]) {
    const map = new Map<Id, IndexedInput>();
    data.forEach(entry => {
      map.set(entry.id, entry);
    });
    this.inputData.set(dataset, map);
  }

  private determineCurrentModelsFromUrl(urlConfiguration: UrlConfiguration) {
    const urlSelectedModels = urlConfiguration.selectedModels;
    const availableModels = Object.keys(this.metadata?.models || {});

    let models: string[] = [];
    if (urlSelectedModels.length > 0) {
      models =
          urlSelectedModels.filter(model => availableModels.includes(model));
    }
    return models.length > 0 ? models : availableModels.slice(0, 1);
  }

  private determineCurrentDatasetFromUrl(urlConfiguration: UrlConfiguration) {
    const urlSelectedDataset = urlConfiguration.selectedDataset || '';

    // Ensure that the currentDataset is part of the available datasets for
    // the currentModel
    const availableDatasets = new Set<string>();
    for (const model of this.currentModels) {
      const modelDatasets = this.metadata?.models?.[model].datasets || [];
      for (const dataset of modelDatasets) {
        availableDatasets.add(dataset);
      }
    }

    if (availableDatasets.has(urlSelectedDataset)) {
      return urlSelectedDataset;
    } else {
      if (availableDatasets.size === 0) {
        this.statusService.addError('No dataset available for loaded models.');
        return '';
      }
      return [...availableDatasets][0];
    }
  }

  setUrlConfiguration(urlConfiguration: UrlConfiguration) {
    this.urlConfiguration = {...urlConfiguration};
  }
}
