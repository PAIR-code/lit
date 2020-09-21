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
import {action, computed, observable, reaction} from 'mobx';

import {LitName, IndexedInput, ServiceUser} from '../lib/types';
import {arrayContainsSame, findSpecKeys} from '../lib/utils';
import {LitService} from './lit_service';
import {AppState, SelectionService, RegressionService, ClassificationService} from './services';

export type Source = {
  modelName: string,
  specKey: LitName,
  fieldName: string
};

type ScoreReader = (id: string) => number | undefined;

export type DeltaInfo = {
  generationKeys: string[]
  deltaRows: DeltaRow[]
};

export type DeltaRow = {
  before?: number,
  after?: number,
  delta?: number,
  d: IndexedInput,
  parent: IndexedInput
};


/**
 * A singleton service for computing deltas between example data points
 * and perturbations.
 */
export class DeltasService extends LitService {
  constructor(
    private readonly appState: AppState,
    private readonly selectionService: SelectionService,
    private readonly classificationService: ClassificationService,
    private readonly regressionService: RegressionService) {
    super();
  }

  /* Get a list of sources for where to read values for deltas from (eg, which
   * fields of the output spec match regression or multiclass prediction, after
   * considering 0/1 multiclass predictions as a single source).
   */
  @computed
  get sources(): Source[] {
    return this.appState.currentModels.flatMap((modelName: string): Source[] => {
      const modelSpec = this.appState.getModelSpec(modelName);
      const outputSpecKeys: LitName[] = ['RegressionScore', 'MulticlassPreds'];
      return outputSpecKeys.flatMap(specKey => {
        const fieldNames = findSpecKeys(modelSpec.output, [specKey]);
        return fieldNames.map(fieldName => ({modelName, specKey, fieldName}));
       });
    });
  }

  // Get a list of each time a generator was run, and the data points generated
  public  deltaInfoFromSource(source: Source): DeltaInfo {
    const byGeneration: {[generationKey: string]: IndexedInput[]} = {};
    this.appState.generatedDataPoints.forEach((d: IndexedInput) => {
      const key = d.meta.creationId;
      byGeneration[key] = (byGeneration[key] || []).concat([d]);
    });

    const deltaRows = Object.keys(byGeneration).flatMap(generationKey => {
      const ds = byGeneration[generationKey];
      return this.deltaRowsForSource(source, ds)
    });
    return {
      generationKeys: Object.keys(byGeneration),
      deltaRows
    };
  }

  private deltaRowsForSource(source: Source, ds: IndexedInput[]): DeltaRow[] {
    const scoreReaders = this.getScoreReaders(source);
    return scoreReaders.flatMap(scoreReader => {
      return this.readDeltaRows(ds, source, scoreReader);
    });
  }

  private readDeltaRows(ds: IndexedInput[], source: Source, readScore: ScoreReader): DeltaRow[] {
    return ds.flatMap(d => {
      const parent = this.appState.getCurrentInputDataById(d.meta.parentId);
      if (parent == null) return [];
      
      const before = readScore(parent.id);
      const after = readScore(d.id);
      const delta = (before != null && after != null)
        ? after - before
        : undefined;
      const deltaRow: DeltaRow = {before,after,delta,d,parent};
      return [deltaRow];
    });
  }

  private getScoreReaders(source: Source): ScoreReader[] {
    const {modelName, specKey, fieldName} = source;

    // Check for regression scores
    if (specKey === 'RegressionScore') {
      const readScoreForRegression: ScoreReader = id => {
        return this.regressionService.regressionInfo[id]?.[modelName]?.[fieldName]?.prediction;
      };
      return [readScoreForRegression];
    }

    // Also support multiclass for multiple classes or binary
    if (specKey === 'MulticlassPreds') {
      const spec = this.appState.getModelSpec(modelName);
      const predictionLabels = spec.output[fieldName].vocab!;
      const margins = this.classificationService.marginSettings[modelName] || {};

      const nullIdx = spec.output[fieldName].null_idx;
      if (predictionLabels.length === 2 && nullIdx != null) {
         const readScoreForMultiClassBinary: ScoreReader = id => {
           return this.classificationService.classificationInfo[id]?.[modelName]?.[fieldName]?.predictions[1 - nullIdx];
        };
        return [readScoreForMultiClassBinary];
      }

      // Multiple classes for multiple tables.
      predictionLabels.map((predictionLabel, index) => {
        const readScoreForMultipleClasses: ScoreReader = id => {
           return this.classificationService.classificationInfo[id]?.[modelName]?.[fieldName]?.predictions[index];
        };
        return readScoreForMultipleClasses;
      });
    }

    // should never reach
    return [];
  }
}
