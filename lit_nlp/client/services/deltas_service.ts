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
import {findSpecKeys} from '../lib/utils';
import {LitService} from './lit_service';
import {AppState, SelectionService, RegressionService, ClassificationService} from './services';

type OutputField = {
  specKey: LitName,
  fieldName: string
};

export type Source = {
  modelName: string,
  specKey: LitName,
  fieldName: string,
  readScoreFn: ScoreReader,
  predictionLabel?: string
};

type ScoreReader = (id: string) => number | undefined;

export type DeltaRow = {
  before?: number,
  after?: number,
  delta?: number,
  d: IndexedInput,
  parent: IndexedInput
};

type GeneratorInterpretation = {
  generationKey: string,
  ruleText: string
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

  /* For grouping outputs from a generator or describing them in user-facing terms */
  public interpretGenerator(d: IndexedInput): GeneratorInterpretation {
    const generationKey = d.meta.creationId;
    const ruleText = d.meta.rule ? d.meta.rule : d.meta.source;
    return {generationKey, ruleText};
  }

  /* Filter by selection */
  public selectedDeltaRows(deltaRows: DeltaRow[]): DeltaRow[] {
    return deltaRows.filter(deltaRow => {
      return this.selectionService.isIdSelected(deltaRow.d.id);
    });
  }
      
  /* Get a list of sources for where to read values for deltas from (eg, which
   * fields of the output spec match regression or multiclass prediction, after
   * considering 0/1 multiclass predictions as a single source).
   */
  public sourcesForModel(modelName: string): Source[] {
    return this.findOutputFields(modelName).flatMap(outputField => {
      return this.createSourcesFor(modelName, outputField);
    });
  }

  private findOutputFields(modelName: string): OutputField[] {
    const modelSpec = this.appState.getModelSpec(modelName);
    const outputSpecKeys: LitName[] = ['RegressionScore', 'MulticlassPreds'];
    return outputSpecKeys.flatMap(specKey => {
      const fieldNames = findSpecKeys(modelSpec.output, [specKey]);
      return fieldNames.map(fieldName => ({fieldName, specKey}));
    });
  }

  private createSourcesFor(modelName: string, outputField: OutputField): Source[] {
    const {specKey, fieldName} = outputField;
    const createSource = (readScoreFn: ScoreReader, predictionLabel?: string) => {
      return {
        ...outputField, 
        modelName,
        readScoreFn,
        predictionLabel
      };
    };

    // Check for regression scores
    if (specKey === 'RegressionScore') {
      const readScoreForRegression: ScoreReader = id => {
        return this.regressionService.regressionInfo[id]?.[modelName]?.[fieldName]?.prediction;
      };
      return [createSource(readScoreForRegression)];
    }

    // Also support multiclass predictions
    if (specKey === 'MulticlassPreds') {
      const spec = this.appState.getModelSpec(modelName);
      const predictionLabels = spec.output[fieldName].vocab!;
      const margins = this.classificationService.marginSettings[modelName] || {};

      // This is really just binary classification
      const nullIdx = spec.output[fieldName].null_idx;
      if (predictionLabels.length === 2 && nullIdx != null) {
         const readScoreForMultiClassBinary: ScoreReader = id => {
           return this.classificationService.classificationInfo[id]?.[modelName]?.[fieldName]?.predictions[1 - nullIdx];
        };
        return [createSource(readScoreForMultiClassBinary)];
      }

      // Multiple classes => one source each
      return predictionLabels.map((predictionLabel, index) => {
        const readScoreForMultipleClasses: ScoreReader = id => {
           return this.classificationService.classificationInfo[id]?.[modelName]?.[fieldName]?.predictions[index];
        };
        return createSource(readScoreForMultipleClasses, predictionLabel);
      });
    }

    // should never reach
    return [];
  }

  /* Get each DeltaRow for a source */
  public readDeltaRowsForSource(source: Source): DeltaRow[] {
    const {readScoreFn} = source;
    const ds = this.appState.generatedDataPoints;
    return ds.flatMap(d => this.readDeltaRows(d, readScoreFn));
  }

  /* For each datapoint, read the score and form a DeltaRow */
  private readDeltaRows(d: IndexedInput, readScore: ScoreReader): DeltaRow[] {
    const parent = this.appState.getCurrentInputDataById(d.meta.parentId);
    if (parent == null) return [];
    
    const before = readScore(parent.id);
    const after = readScore(d.id);
    const delta = (before != null && after != null)
      ? after - before
      : undefined;
    const deltaRow: DeltaRow = {before,after,delta,d,parent};
    return [deltaRow];
  }
}
