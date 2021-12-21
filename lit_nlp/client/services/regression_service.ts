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
import * as d3 from 'd3';
import {computed, observable, reaction} from 'mobx';

import {ColorOption, D3Scale, IndexedInput, Preds} from '../lib/types';
import {findSpecKeys} from '../lib/utils';
import {
  CONTINUOUS_SIGNED, CONTINUOUS_UNSIGNED, MULTIHUE_CONTINUOUS
} from '../lib/colors';

import {LitService} from './lit_service';
import {ApiService, AppState} from './services';

interface AllRegressionInfo {
  [id: string]: PerExampleRegressionInfo;
}

interface PerExampleRegressionInfo {
  [model: string]: PerExamplePerModelRegressionInfo;
}

interface PerExamplePerModelRegressionInfo {
  [predKey: string]: RegressionInfo;
}

/**
 * Info about individual regressions including computed properties.
 * These interface field names should be in sync with the
 * fieldsToDisplayNames Map.
 */
export interface RegressionInfo {
  prediction: number;
  error: number;
  squaredError: number;
}

/**
 * A map from regression info field names to their display names.
 */
const regressionDisplayNames = new Map([
  ['prediction', 'prediction'],
  ['error', 'error'],
  ['squaredError', 'squared error'],
]);

/**
 * Stores min and max values for all computed info.
 */
interface InfoRanges {
  prediction: [number, number];
  error: [number, number];
  squaredError: [number, number];
}

/**
 * A singleton class that handles calculating and storing per-input
 * regression response information.
 */
export class RegressionService extends LitService {
  @observable regressionInfo: AllRegressionInfo = {};

  constructor(
      private readonly apiService: ApiService,
      private readonly appState: AppState) {
    super();
    reaction(() => appState.currentModels, currentModels => {
      this.reset();
    });
  }

  /**
   * Calls the server to get regression predictions and calculate related info.
   * @param inputs inputs to run model on
   * @param model model to query
   * @param datasetName current dataset (for caching)
   */
  async getRegressionPreds(
      inputs: IndexedInput[], model: string,
      datasetName: string): Promise<Preds[]> {
    // TODO(lit-team): Use client-side cache when available.
    const result = this.apiService.getPreds(
        inputs, model, datasetName, ['RegressionScore']);
    const preds = await result;
    if (preds != null && preds.length > 0) {
      this.processNewPreds(inputs, model, preds);
    }
    return result;
  }

  /**
   * Reset stored info. Used when active models change.
   */
  reset() {
    this.regressionInfo = {};
  }

  /**
   * Gets stored results for a given datapoint and prediction.
   */
  async getResults(ids: string[], model: string, predKey: string):
      Promise<RegressionInfo[]> {
    const unstoredIds: string[] = [];

    ids.forEach((id) => {
      if (this.regressionInfo[ids[0]]?.[model]?.[predKey] == null) {
        unstoredIds.push(id);
      }
    });

    // If any results aren't yet stored in the front-end, then get them.
    if (unstoredIds.length > 0) {
      await this.getRegressionPreds(
          this.appState.getExamplesById(unstoredIds), model,
          this.appState.currentDataset);
    }

    const results: RegressionInfo[] = [];
    ids.forEach((id) => {
      const regressionInfo = this.regressionInfo[id]?.[model]?.[predKey];
      if (regressionInfo != null) {
        results.push(regressionInfo);
      }
    });
    return results;
  }

  private processNewPreds(
      inputs: IndexedInput[], model: string, preds: Preds[]) {
    const outputSpec = this.appState.currentModelSpecs[model].spec.output;
    const regressionKeys = findSpecKeys(outputSpec, 'RegressionScore');
    const predictedKeys = Object.keys(preds[0]);

    for (let i = 0; i < preds.length; i++) {
      const input = inputs[i];
      const pred = preds[i];
      if (this.regressionInfo[input.id] == null) {
        this.regressionInfo[input.id] = {} as PerExampleRegressionInfo;
      }
      if (this.regressionInfo[input.id][model] == null) {
        this.regressionInfo[input.id][model] = {} as
            PerExamplePerModelRegressionInfo;
      }
      for (let predIndex = 0; predIndex < predictedKeys.length; predIndex++) {
        const predKey = predictedKeys[predIndex];
        if (!regressionKeys.includes(predKey)) {
          continue;
        }
        // Set regression info for the prediction.
        const info = {} as RegressionInfo;
        info.prediction = pred[predKey] as number;

        const labelField = outputSpec[predKey].parent;
        if (labelField != null && inputs[i].data[labelField] != null) {
          info.error = info.prediction - (+inputs[i].data[labelField]);
          info.squaredError = info.error * info.error;
        }
        if (this.regressionInfo[input.id][model][predKey] == null) {
          this.regressionInfo[input.id][model][predKey] = info;
        }
      }
    }
  }

  /**
   * Loops through all stored info to get min/max ranges for that info.
   */
  @computed
  get ranges(): {[modelAndKey: string]: InfoRanges} {
    const ranges: {[modelAndKey: string]: InfoRanges} = {};
    Object.values(this.regressionInfo)
        .forEach((modelInfos: PerExampleRegressionInfo) => {
          const models = Object.keys(modelInfos);
          models.forEach((model: string) => {
            const predKeys = Object.keys(modelInfos[model]);
            predKeys.forEach((predKey: string) => {
              const info = modelInfos[model][predKey];
              const rangesKey = `${model}:${predKey}`;
              if (ranges[rangesKey] == null) {
                ranges[rangesKey] = {
                  prediction: [Infinity, -Infinity],
                  error: [Infinity, -Infinity],
                  squaredError: [Infinity, -Infinity]
                };
              }
              ranges[rangesKey].prediction[0] =
                  Math.min(ranges[rangesKey].prediction[0], info.prediction);
              ranges[rangesKey].prediction[1] =
                  Math.max(ranges[rangesKey].prediction[1], info.prediction);
              ranges[rangesKey].error[0] =
                  Math.min(ranges[rangesKey].error[0], info.error);
              ranges[rangesKey].error[1] =
                  Math.max(ranges[rangesKey].error[1], info.error);
              ranges[rangesKey].squaredError[0] = Math.min(
                  ranges[rangesKey].squaredError[0], info.squaredError);
              ranges[rangesKey].squaredError[1] = Math.max(
                  ranges[rangesKey].squaredError[1], info.squaredError);

              ranges[rangesKey].prediction =
                  this.adjustDomain(...ranges[rangesKey].prediction);
              ranges[rangesKey].error =
                  this.adjustDomain(...ranges[rangesKey].error);
              ranges[rangesKey].squaredError =
                  this.adjustDomain(...ranges[rangesKey].squaredError);
            });
          });
        });
    return ranges;
  }

  @computed
  get colorOptions(): ColorOption[] {
    const ids = Object.keys(this.regressionInfo);
    if (ids.length === 0) {
      return [];
    }
    const options: ColorOption[] = [];
    const info = this.regressionInfo[ids[0]];
    const models = Object.keys(info);
    for (const model of models) {
      const predKeys = Object.keys(info[model]);
      for (const predKey of predKeys) {
        const predDomain = this.ranges[`${model}:${predKey}`].prediction;
        options.push({
          name: `${model}:${predKey} prediction`,
          getValue: (input: IndexedInput) =>
              this.regressionInfo[input.id][model][predKey].prediction,
          scale: d3.scaleSequential(MULTIHUE_CONTINUOUS)
                   .domain(predDomain) as D3Scale
        });
        const errAbsMax:number = Math.max(
          ...this.ranges[`${model}:${predKey}`].error.map(v => Math.abs(v)));
        const errDomain: [number, number] = [-errAbsMax, errAbsMax];
        options.push({
          name: `${model}:${predKey} error`,
          getValue: (input: IndexedInput) =>
              this.regressionInfo[input.id][model][predKey].error,
          scale: d3.scaleSequential(CONTINUOUS_SIGNED)
                   .domain(errDomain) as D3Scale
        });
        const sqErrMax =
          Math.max(...this.ranges[`${model}:${predKey}`].squaredError);
        const sqErrDomain: [number, number] = [0, sqErrMax];
        options.push({
          name: `${model}:${predKey} squared error`,
          getValue: (input: IndexedInput) =>
              this.regressionInfo[input.id][model][predKey].squaredError,
          scale: d3.scaleSequential(CONTINUOUS_UNSIGNED)
                   .domain(sqErrDomain) as D3Scale
        });
      }
    }
    return options;
  }

  /**
   * Returns a key for prediction error.
   */
  getErrorKey(key: string) {
    return `${key}:error`;
  }

  getInfoFields() {
    return Array.from(regressionDisplayNames.keys());
  }

  getDisplayNames() {
    return Array.from(regressionDisplayNames.values());
  }

  private adjustDomain(min: number, max: number): [number, number] {
    // If domain has a negative min and positive max, then make the domain
    // symmetric around 0.
    if (min * max >= 0) {
      return [min, max];
    } else {
      const largest = Math.max(-min, max);
      return [-largest, largest];
    }
  }
}
