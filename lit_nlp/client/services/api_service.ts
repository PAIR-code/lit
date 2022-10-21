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

import {LitTypeTypesList} from '../lib/lit_types';
import {CallConfig, IndexedInput, LitMetadata, Preds} from '../lib/types';
import {deserializeLitTypesInLitMetadata, getTypeNames} from '../lib/utils';

import {LitService} from './lit_service';
import {StatusService} from './status_service';


/**
 * API service singleton, responsible for actually making calls to the server
 * and (as best it can) enforcing type safety on returned values.
 */
export class ApiService extends LitService {
  constructor(private readonly statusService: StatusService) {
    super();
  }

  /**
   * Send a request to the server to get inputs for a dataset.
   * @param dataset name of dataset to load
   */
  async getDataset(dataset: string): Promise<IndexedInput[]> {
    const loadMessage = 'Loading inputs';
    const examples = await this.queryServer<IndexedInput[]>(
        '/get_dataset', {'dataset_name': dataset}, [], loadMessage);
    if (examples == null) {
      const errorText = 'Failed to load dataset (server returned null).';
      this.statusService.addError(errorText);
      throw (new Error(errorText));
    }
    return examples;
  }

  /**
   * Request the server to create a new dataset.
   * Loads the server on the backend, but examples need to be
   * fetched to the frontend separately using getDataset().
   * Returns (updated metadata, name of just-loaded dataset)
   * @param dataset name of (base) dataset to dispatch to load()
   * @param datasetPath path to load from
   */
  async createDataset(dataset: string, datasetPath: string):
      Promise<[LitMetadata, string]> {
    const loadMessage = 'Creating new dataset';
    return this.queryServer(
        '/create_dataset',
        {'dataset_name': dataset, 'dataset_path': datasetPath},
        [], loadMessage);
  }

  /**
   * Request the server to create a new model.
   * Loads the model on the backend.
   * Returns (updated metadata, name of just-loaded model)
   * @param model name of (base) model to dispatch to load()
   * @param modelPath path to load from
   */
  async createModel(model: string, modelPath: string):
      Promise<[LitMetadata, string]> {
    const loadMessage = 'Loading new model';
    return this.queryServer(
        '/create_model',
        {'model_name': model, 'model_path': modelPath},
        [], loadMessage);
  }

  /**
   * Send a request to the server to get dataset info.
   */
  async getInfo(): Promise<LitMetadata> {
    const loadMessage = 'Loading metadata';
    return this.queryServer<LitMetadata>('/get_info', {}, [], loadMessage)
        .then((metadata) => deserializeLitTypesInLitMetadata(metadata));
  }

  /**
   * Calls the server to get predictions of the given types.
   * @param inputs inputs to run model on
   * @param model model to query
   * @param datasetName current dataset (for caching)
   * @param requestedTypes datatypes to request
   * @param requestedFields optional fields to request
   * @param loadMessage optional loading message to display in toolbar
   */
  getPreds(
      inputs: IndexedInput[], model: string, datasetName: string,
      requestedTypes: LitTypeTypesList, requestedFields?: string[],
      loadMessage?: string): Promise<Preds[]> {
    loadMessage = loadMessage || 'Fetching predictions';
    requestedFields = requestedFields || [];
    return this.queryServer(
        '/get_preds', {
          'model': model,
          'dataset_name': datasetName,
          'requested_types': getTypeNames(requestedTypes).join(','),
          'requested_fields': requestedFields.join(','),
        },
        inputs, loadMessage);
  }

  /**
   * Calls the server to get newly generated inputs for a set of inputs, for a
   * given generator and model.
   * @param inputs inputs to run on
   * @param modelName model to query
   * @param datasetName current dataset
   * @param generator generator being used
   * @param config: configuration to send to backend (optional)
   * @param loadMessage: loading message to show to user (optional)
   */
  async getGenerated(
      inputs: IndexedInput[], modelName: string, datasetName: string,
      generator: string, config?: CallConfig,
      loadMessage?: string): Promise<IndexedInput[][]> {
    loadMessage = loadMessage ?? 'Loading generator output';
    return this.queryServer<IndexedInput[][]>(
        '/get_generated', {
          'model': modelName,
          'dataset_name': datasetName,
          'generator': generator,
        },
        inputs, loadMessage, config);
  }

  /**
   * Calls the server to create and set the IDs and other data for the provided
   * inputs.
   * @param inputs Inputs to get the IDs for.
   * @param datasetName current dataset
   *
   * @return Inputs with the IDs correctly set.
   */
  annotateNewData(
      inputs: IndexedInput[], datasetName: string): Promise<IndexedInput[]> {
    return this.queryServer<IndexedInput[]>(
        '/annotate_new_data', {
          'dataset_name': datasetName,
        },
        inputs);
  }

  fetchNewData(savedDatapointsId: string): Promise<IndexedInput[]> {
    return this.queryServer<IndexedInput[]>(
        '/fetch_new_data', {
          'saved_datapoints_id': savedDatapointsId,
        },
        []);
  }

  /**
   * Calls the server to run an interpretation component.
   * @param inputs inputs to run on
   * @param modelName model to query
   * @param datasetName current dataset (for caching)
   * @param interpreterName interpreter to run
   * @param config: configuration to send to backend (optional)
   * @param loadMessage: loading message to show to user (optional)
   */
  getInterpretations(
      inputs: IndexedInput[], modelName: string, datasetName: string,
      interpreterName: string, config?: CallConfig,
      // tslint:disable-next-line:no-any
      loadMessage?: string): Promise<any> {
    loadMessage = loadMessage ?? 'Fetching interpretations';
    return this.queryServer(
        '/get_interpretations', {
          'model': modelName,
          'dataset_name': datasetName,
          'interpreter': interpreterName,
        },
        inputs, loadMessage, config);
  }

  /**
   * Calls the server to save new datapoints.
   * @param inputs Text inputs to persist.
   * @param datasetName dataset being used.
   * @param path path to save to.
   */
  saveDatapoints(inputs: IndexedInput[], datasetName: string, path: string):
      Promise<string> {
    const loadMessage = 'Saving new datapoints';
    return this.queryServer(
        '/save_datapoints', {
          'dataset_name': datasetName,
          path,
        },
        inputs, loadMessage);
  }

  /**
   * Calls the server to load persisted datapoints.
   * @param datasetName dataset being used,
   * @param path path to load from.
   */
  loadDatapoints(datasetName: string, path: string): Promise<IndexedInput[]> {
    const loadMessage = 'Loading new datapoints';
    return this.queryServer(
        '/load_datapoints', {
          'dataset_name': datasetName,
          path,
        },
        [], loadMessage);
  }

  /**
   * Push UI state to the server.
   * @param selection current selection
   * @param datasetName dataset being used,
   * @param config additional params to pass to UIStateTracker.update_state()
   */
  pushState(
      selection: IndexedInput[], datasetName: string, config?: CallConfig) {
    const loadMessage = 'Syncing UI state.';
    return this.queryServer(
        '/push_ui_state', {'dataset_name': datasetName}, selection, loadMessage,
        config);
  }

  /**
   * Send a standard request to the server.
   * @param endpoint server endpoint, like /get_preds
   * @param params query params
   * @param inputs input examples
   */
  private async queryServer<T>(
      endpoint: string, params: {[key: string]: string}, inputs: IndexedInput[],
      loadMessage: string = '', config?: CallConfig): Promise<T> {
    const finished = this.statusService.startLoading(loadMessage);
    // For a smaller request, replace known (original) examples with their IDs;
    // we can simply look these up on the server.
    // TODO: consider sending the metadata as well, since this might be changed
    // from the frontend.
    const processedInputs: Array<IndexedInput|string> = inputs.map(input => {
      if (!input.meta['added']) {
        return input.id;
      }
      return input;
    });

    const paramsArray =
        Object.keys(params).map((key: string) => `${key}=${params[key]}`);
    const url = encodeURI(`.${endpoint}?${paramsArray.join('&')}`);
    const body = JSON.stringify({inputs: processedInputs, config});
    try {
      const res = await fetch(url, {method: 'POST', body});
      // If there is tsserver error, the response contains text (not json).
      if (!res.ok) {
        const text = await res.text();
        throw (new Error(text));
      }
      const json = await res.json();
      finished();
      // When a call finishes, clear any previous error of the same call.
      this.statusService.removeError(url);
      return json;
      // tslint:disable-next-line:no-any
    } catch (err: any) {
      finished();
      // Extract error text if returned from tsserver.
      const found = err.message.match('^.*?(?=\n\nDetails:)');
      if (!found) {
        this.statusService.addError(
            'Unknown error', err.toString(), url);
      } else {
        this.statusService.addError(found[0], err.toString(), url);
      }
      throw (err);
    }
  }
}
