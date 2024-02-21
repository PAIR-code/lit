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

import {FieldMatcher, ImageBytes} from '../lib/lit_types';
import {defaultValueByField, IndexedInput, Input, type LitCanonicalLayout, type LitComponentLayouts, type LitMetadata, ModelInfo, type ModelInfoMap, ModelSpec, NONE_DS_DICT_KEY, type Spec} from '../lib/types';
import {findSpecKeys, getTypes} from '../lib/utils';

import {ApiService} from './api_service';
import {LitService} from './lit_service';
import {StatusService} from './services';
import {StateObservedByUrlService, UrlConfiguration} from './url_service';


type Id = string;
type DatasetName = string;
type IndexedInputMap = Map<Id, IndexedInput>;

/** Function type to get callbacks for newly added datapoints. **/
export type NewDatapointsFn = (datapoints: IndexedInput[]) => void;

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

  @observable documentationOpen = false;
  // TODO(b/204677206): While cleaning up console warnings, find a better way to
  // initialize the app so that we don't need this non-null assertion here
  // https://www.typescriptlang.org/docs/handbook/release-notes/typescript-2-0.html#non-null-assertion-operator
  @observable metadata!: LitMetadata;
  @observable currentModels: string[] = [];
  @observable compareExamplesEnabled = false;
  @observable layoutName!: string;
  @observable layouts: LitComponentLayouts = {};
  private readonly newDatapointsCallbacks: NewDatapointsFn[] = [];

  @computed
  get layout(): LitCanonicalLayout {
    return this.layouts[this.layoutName];
  }

  /**
   * Enforce setting currentDataset through the setCurrentDataset method by
   * making the currentDatasetInternal private...
   */
  @observable private currentDatasetInternal = '';
  @computed
  get currentDataset(): string {
    return this.currentDatasetInternal;
  }

  /**
   * Set current dataset.
   */
  @action
  setCurrentDataset(dataset: string) {
    if (!this.inputData.has(dataset)) {
      throw new Error(
          `Dataset '${dataset}' is not loaded. Call loadDataset() first.`);
    }
    this.currentDatasetInternal = dataset;
  }

  /**
   * Set dataset with given examples. Only use this in tests.
   * TODO(b/297232000): get rid of test-only methods, have some mock init
   * instead.
   */
  @action
  setDatasetForTest(dataset: string, dataMap: Map<string, IndexedInput>) {
    this.inputData.set(dataset, dataMap);  // simulates a call to loadDataset()
    this.setCurrentDataset(dataset);
  }

  @computed
  get currentDatasetSpec(): Spec {
    return this.metadata.datasets[this.currentDataset].spec;
  }

  @computed
  get datasetHasImages(): boolean {
    return findSpecKeys(this.currentDatasetSpec, ImageBytes).length > 0;
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
    Object.values(this.currentModelInfos).forEach((modelInfo: ModelInfo) => {
      Object.keys(modelInfo.spec.input).forEach(key => {
        if (modelInfo.spec.input[key].required === true) {
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
  getAncestry(id?: string): string[] {
    const ret: string[] = [];
    while (id) {
      ret.push(id);
      id = this.getCurrentInputDataById(id)?.meta.parentId;
    }
    return ret;
  }

  /**
   * Select models.
   */
  @action
  setCurrentModels(currentModels: string[]) {
    this.currentModels = currentModels;
  }

  /**
   * Get the configs of only the current models.
   */
  @computed
  get currentModelInfos(): ModelInfoMap {
    const {currentModels} = this;
    return Object.entries(this.metadata.models).reduce(
        (currentModelInfos: ModelInfoMap, [modelName, modelInfo]) => {
          if (currentModels.includes(modelName)) {
            currentModelInfos[modelName] = modelInfo;
          }
          return currentModelInfos;
        }, {});
  }

  /**
   * Get the input and output spec for a particular model.
   */
  getModelSpec(modelName: string): ModelSpec {
    const modelInfo = this.metadata.models[modelName];
    if (modelInfo != null) {
      return modelInfo.spec;
    } else {
      throw new Error(`Model ${modelName} not found in LitMetadata.`);
    }
  }

  /**
   * Get the spec keys matching the info from the provided FieldMatcher.
   */
  getSpecKeysFromFieldMatcher(matcher: FieldMatcher, modelName: string) {
    let spec = this.currentDatasetSpec;
    if (matcher.spec === 'output') {
      spec = this.getModelSpec(modelName).output;
    } else if (matcher.spec === 'input') {
      spec = this.getModelSpec(modelName).input;
    }
    return findSpecKeys(spec, getTypes(matcher.types));
  }

  //=================================== Generation logic
  /**
   * Create an empty datapoint with appropriate default values for each field.
   */
  makeEmptyDatapoint(source?: string) {
    const data: Input = {'_id': '', '_meta': {source, added: true}};
    const spec = this.currentDatasetSpec;
    for (const key of this.currentInputDataKeys) {
      data[key] = defaultValueByField(key, spec);
    }
    return {data, id: '', meta: data['_meta']};
  }

  /**
   * Annotate one or more bare datapoints.
   * @param data input examples; ids will be overwritten.
   */
  async annotateNewData(data: IndexedInput[]): Promise<IndexedInput[]> {
    // Legacy: this exists as a pass-through so lit_app.ts and url_service.ts
    // don't need to depend on the ApiService directly.
    return this.apiService.annotateNewData(data, this.currentDataset);
  }

  addNewDatapointsCallback(callback: NewDatapointsFn) {
    this.newDatapointsCallbacks.push(callback);
  }

  /**
   * Atomically commit new datapoints to the active dataset.
   * Note that (currently) this state is entirely stored on the frontend;
   * if the page is reloaded the newly-added points will not be there unless
   * recovered via URL params or another mechanism.
   */
  @action
  commitNewDatapoints(datapoints: IndexedInput[]) {
    const committedDatapoints: IndexedInput[] = [];
    for (const entry of datapoints) {
      // If the new datapoint already exists in the input data, do not overwrite
      // it with this new copy, as that will cause issues with datapoint parent
      // tracking (an infinite loop of parent pointers).
      if (this.currentInputDataById.has(entry.id)) {
        console.log(
            'Attempted to add existing datapoint, ignoring add request.',
            toJS(entry));
      } else {
        this.currentInputDataById.set(entry.id, entry);
        committedDatapoints.push(entry);
      }
    }
    for (const callback of this.newDatapointsCallbacks) {
      callback(datapoints);
    }
  }


  //=================================== Initialization logic
  addLayouts(layouts: LitComponentLayouts) {
    Object.assign(this.layouts, layouts);
  }

  @action
  async initialize() {
    // TODO(b/160480922) Move away from AppState being the source of truth for
    // URL configuration data.
    const {urlConfiguration} = this;
    console.log('[LIT - url configuration]', urlConfiguration);
    // Add any custom layouts that were specified in Python.
    this.addLayouts(this.metadata.layouts);
    this.layoutName = urlConfiguration.layoutName || this.metadata.defaultLayout;

    // TODO(b/160480922) Move away from AppState being the source of truth for
    // URL configuration data.
    this.currentModels = this.determineCurrentModelsFromUrl(urlConfiguration);
    // This is async because it may trigger the backend to load data from disk.
    const dataset = await this.determineCurrentDatasetFromUrl(urlConfiguration);
    await this.loadDataset(dataset);
    this.setCurrentDataset(dataset);

    this.initialized = true;
  }

  async loadDataset(dataset: string) {
    if (!dataset) return;
    if (this.inputData.has(dataset)) return;

    const response = await this.apiService.getDataset(dataset);
    const map = new Map<Id, IndexedInput>();
    response.forEach(entry => {
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

  private async determineCurrentDatasetFromUrl(urlConfiguration: UrlConfiguration) {
    const urlSelectedDataset = urlConfiguration.selectedDataset || '';
    const urlNewDatasetPath = urlConfiguration.newDatasetPath;
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
      // If the url param is set for creating a new dataset from a path, try
      // to do that.
      let newlyCreatedDataset;
      if (urlNewDatasetPath) {
        newlyCreatedDataset = await this.createNewDataset(
          urlSelectedDataset, urlNewDatasetPath);
        // If the dataset was successfully created, select it.
        if (newlyCreatedDataset) {
          return newlyCreatedDataset;
        }
      }
      // Return the selected dataset.
      return urlSelectedDataset;
    }

    // If the dataset is not compatible with the selected models, return the
    // first compatible dataset.
    else {
      if (availableDatasets.size === 0) {
        console.log(
            'No dataset available for loaded models, using the empty dataset.');
        return NONE_DS_DICT_KEY;
      }
      return [...availableDatasets][0];
    }
  }

  /**
   * Try to create a new dataset, if the url param is set.
   * If the url param is not set, or if the dataset creation fails, return null.
   * @param urlSelectedDataset Original dataset (to clone from).
   * @param urlNewDatasetPath Path of new datasest.
   */
  private async createNewDataset(
    urlSelectedDataset: string,
    urlNewDatasetPath: string){
    try {
      const newInfo = await this.apiService.createDataset(
        urlSelectedDataset, {});
      this.metadata = newInfo[0];
      return newInfo[1];
    } catch {
      this.statusService.addError(`Could not load dataset from
        ${urlNewDatasetPath}. See console for more details.`);
      return;
    }
  }

  setUrlConfiguration(urlConfiguration: UrlConfiguration) {
    this.urlConfiguration = {...urlConfiguration};
  }
  getUrlConfiguration() {
    return this.urlConfiguration;
  }

  /**
   * Get best URL for this server.
   */
  getBestURL() {
    let urlBase = (this.metadata.canonicalURL || window.location.origin);
    return new URL(`${urlBase}${window.location.search}`).href;
  }
}
