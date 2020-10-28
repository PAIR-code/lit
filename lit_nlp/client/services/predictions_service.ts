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
import {LitName, IndexedInput} from '../lib/types';
import {Preds} from '../lib/types';
import { ApiService } from './api_service';
import {LitService} from './lit_service';
import {AppState, SelectionService, RegressionService, ClassificationService} from './services';

/**
 * A singleton service for fetching predictions across LIT services and output types.
 */
export class PredictionsService extends LitService {
  constructor(
    private readonly appState: AppState,
    private readonly apiService: ApiService,
    private readonly classificationService: ClassificationService,
    private readonly regressionService: RegressionService) {
    super();
  }

  /**
   * Get predictions from the backend for all input data and display by
   * prediction score in the plot.
   */
  public async ensurePredictionsFetched(model: string) {
    const currentInputData = this.appState.currentInputData;
    if (currentInputData == null) {
        return;
    }

    // TODO(lit-dev): consolidate to a single call here, with client-side cache.
    const dataset = this.appState.currentDataset;
    const results = await Promise.all([
        this.classificationService.getClassificationPreds(
            currentInputData, model, dataset),
        this.regressionService.getRegressionPreds(
            currentInputData, model, dataset),
        this.apiService.getPreds(
            currentInputData, model, dataset, ['Scalar']),
    ]);
    if (results === null) {
        return;
    }
    const classificationPreds = results[0];
    const regressionPreds = results[1];
    const scalarPreds = results[2];
    if (classificationPreds == null && regressionPreds == null &&
        scalarPreds == null) {
        return;
    }

    const preds: Preds[] = [];
    for (let i = 0; i < classificationPreds.length; i++) {
        const currId = currentInputData[i].id;
        // TODO(lit-dev): structure this as a proper IndexedInput,
        // rather than having 'id' as a regular field.
        const pred = Object.assign(
            {}, classificationPreds[i], scalarPreds[i], regressionPreds[i],
            {id: currId});
        preds.push(pred);
    }

    // Add the error info for any regression keys.
    if (regressionPreds != null) {
        const ids = currentInputData.map(data => data.id);
        const regressionKeys = Object.keys(regressionPreds[0]);
        for (let j = 0; j < regressionKeys.length; j++) {
          const regressionInfo = await this.regressionService.getResults(
            ids, model, regressionKeys[j]);
          for (let i = 0; i < preds.length; i++) {
            let errorKey = this.regressionService.getErrorKey(regressionKeys[j]);
            preds[i][errorKey] = regressionInfo[i].error;
          }
        }
    }

    return preds;
  }
}
