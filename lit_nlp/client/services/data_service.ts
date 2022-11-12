/**
 * @license
 * Copyright 2022 Google LLC
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

import {BINARY_NEG_POS, ColorRange} from '../lib/colors';
import {BooleanLitType, CategoryLabel, GeneratedText, GeneratedTextCandidates, LitType, MulticlassPreds, RegressionScore, Scalar} from '../lib/lit_types';
import {ClassificationResults, IndexedInput, RegressionResults} from '../lib/types';
import {createLitType, findSpecKeys, isLitSubtype, mapsContainSame} from '../lib/utils';

import {LitService} from './lit_service';
import {ApiService, AppState, ClassificationService, SettingsService} from './services';


/** Data source for a data column. */
export type Source = string;

/** Type for a data value. */
//tslint:disable-next-line:no-any
type ValueType = any;

/** Function type to set a column's data value for a new datapoint. **/
export type ValueFn = (input: IndexedInput) => ValueType;

/** Info about a data column. */
export interface DataColumnHeader {
  dataType: LitType;
  name: string;
  key: string;
  source: Source;
  getValueFn: ValueFn;
  colorRange?: ColorRange;
}

/** Types of columns auto calculated by data service. */
export enum CalculatedColumnType {
  PREDICTED_CLASS = 'class',
  CORRECT = 'correct',
  ERROR = 'error',
  SQUARED_ERROR = 'squared error',
}

/** Map of datapoint ID to values for a column of data. */
export type ColumnData = Map<string, ValueType>;

/** Column source prefix for columns from the classification interpreter. */
export const CLASSIFICATION_SOURCE_PREFIX = 'Classification';
/** Column source prefix for columns from GeneratedText model outputs. */
export const GEN_TEXT_SOURCE_PREFIX = 'GeneratedText';
/** Column source prefix for columns from GeneratedTextCandidates outputs. */
export const GEN_TEXT_CANDS_SOURCE_PREFIX = 'GeneratedTextCandidates';
/** Column source prefix for columns from the regression interpreter. */
export const REGRESSION_SOURCE_PREFIX = 'Regression';
/** Column source prefix for columns from scalar model outputs. */
export const SCALAR_SOURCE_PREFIX = 'Scalar';

/**
 * Data service singleton, responsible for maintaining columns of computed data
 * for datapoints in the current dataset.
 */
export class DataService extends LitService {
  @observable private readonly columnHeaders =
      new Map<string, DataColumnHeader>();
  @observable readonly columnData = new Map<string, ColumnData>();

  constructor(
      private readonly appState: AppState,
      private readonly classificationService: ClassificationService,
      private readonly apiService: ApiService,
      private readonly settingsService: SettingsService) {
    super();
    reaction(() => appState.currentDataset, () => {
      this.columnHeaders.clear();
      this.columnData.clear();
    });

    // Run classification interpreter when the inputs or margins change.
    const getClassificationInputs = () =>
        [this.appState.currentInputData, this.appState.currentModels,
         this.classificationService.allMarginSettings];
    reaction(getClassificationInputs, () => {
      if (this.appState.currentInputData == null ||
          this.appState.currentInputData.length === 0 ||
          this.appState.currentModels.length === 0 ||
          !this.settingsService.isValidCurrentDataAndModels) {
        return;
      }
      for (const model of this.appState.currentModels) {
        this.runClassification(model, this.appState.currentInputData);
      }
    }, {fireImmediately: true});

    // Run other preiction interpreters when necessary.
    const getPredictionInputs =
        () => [this.appState.currentInputData, this.appState.currentModels];
    reaction(getPredictionInputs, () => {
      if (this.appState.currentInputData == null ||
          this.appState.currentInputData.length === 0 ||
          this.appState.currentModels.length === 0 ||
          !this.settingsService.isDatasetValidForModels(
              this.appState.currentDataset, this.appState.currentModels)) {
        return;
      }
      for (const model of this.appState.currentModels) {
        this.runGeneratedTextPreds(model, this.appState.currentInputData);
        this.runRegression(model, this.appState.currentInputData);
        this.runScalarPreds(model, this.appState.currentInputData);
      }
    }, {fireImmediately: true});

    this.appState.addNewDatapointsCallback(async (newDatapoints) =>
      this.setValuesForNewDatapoints(newDatapoints));
  }

  /**
   * Get the column name given a model, key, and optional type.
   *
   * Note that colons are used as separators between these three pieces of data,
   * so none of these values should contain a colon.
   */
  getColumnName(model: string, predKey: string, type?: CalculatedColumnType) {
    return `${model}:${predKey}${type != null ? `:${type}` : ''}`;
  }

  /**
   * Run classification interpreter and store results in data service.
   */
  private async runClassification(model: string, data: IndexedInput[]) {
    const {output} = this.appState.currentModelSpecs[model].spec;
    if (findSpecKeys(output, MulticlassPreds).length === 0) {
      return;
    }

    const interpreterPromise = this.apiService.getInterpretations(
        data, model, this.appState.currentDataset, 'classification',
        this.classificationService.marginSettings[model],
        `Computing classification results`);
    const classificationResults = await interpreterPromise;

    // Add classification results as new columns to the data service.
    if (classificationResults == null || classificationResults.length === 0) {
      return;
    }
    const classificationKeys = Object.keys(classificationResults[0]);
    for (const key of classificationKeys) {
      // Parse results into new data columns and add the columns.
      const scoreFeatName = this.getColumnName(model, key);
      const predClassFeatName =
          this.getColumnName(model, key, CalculatedColumnType.PREDICTED_CLASS);
      const correctnessName =
          this.getColumnName(model, key, CalculatedColumnType.CORRECT);
      const scores = classificationResults.map(
          (result: ClassificationResults) => result[key].scores);
      const predClasses = classificationResults.map(
          (result: ClassificationResults) => result[key].predicted_class);
      const correctness = classificationResults.map(
          (result: ClassificationResults) => result[key].correct);
      const source = `${CLASSIFICATION_SOURCE_PREFIX}:${model}`;
      this.addColumnFromList(
          scores, data, key, scoreFeatName, createLitType(MulticlassPreds),
          source);
      const litTypeClassification = createLitType(CategoryLabel);
      if (output[key] instanceof MulticlassPreds) {
        const predSpec = output[key] as MulticlassPreds;
        litTypeClassification.vocab = predSpec.vocab;
        this.addColumnFromList(
            predClasses, data, key, predClassFeatName, litTypeClassification,
            source);
        if (predSpec.parent != null) {
          this.addColumnFromList(
              correctness, data, key, correctnessName,
              createLitType(BooleanLitType), source, () => null,
              BINARY_NEG_POS);
        }
      }
    }
  }

  private async runGeneratedTextPreds(model: string, data: IndexedInput[]) {
    const genTextTypes = [GeneratedText, GeneratedTextCandidates];
    const {output} = this.appState.currentModelSpecs[model].spec;
    if (findSpecKeys(output, genTextTypes).length === 0) {return;}

    const predsPromise = this.apiService.getPreds(
      data, model, this.appState.currentDataset, genTextTypes);
    const preds = await predsPromise;
    if (preds == null || preds.length === 0) {return;}

    const genTextKeys = Object.keys(preds[0]);
    for (const key of genTextKeys) {
      const isGenText = output[key] instanceof GeneratedText;
      const genTextFeatureName = this.getColumnName(model, key);
      const genText = preds.map((p) => p[key]);
      const dataType =  isGenText ? createLitType(GeneratedText) :
                                    createLitType(GeneratedTextCandidates);
      const source =  `${isGenText ? GEN_TEXT_SOURCE_PREFIX :
                                     GEN_TEXT_CANDS_SOURCE_PREFIX}:${model}`;
      this.addColumnFromList(
          genText, data, key, genTextFeatureName, dataType, source);
    }
  }

  /**
   * Run regression interpreter and store results in data service.
   */
  private async runRegression(model: string, data: IndexedInput[]) {
    const {output} = this.appState.currentModelSpecs[model].spec;
    if (findSpecKeys(output, RegressionScore).length === 0) {
      return;
    }

    const interpreterPromise = this.apiService.getInterpretations(
        data, model, this.appState.currentDataset, 'regression', undefined,
        `Computing regression results`);
    const regressionResults = await interpreterPromise;

    // Add regression results as new columns to the data service.
    if (regressionResults == null || regressionResults.length === 0) {
      return;
    }
    const regressionKeys = Object.keys(regressionResults[0]);
    for (const key of regressionKeys) {
      // Parse results into new data columns and add the columns.
      const scoreFeatName = this.getColumnName(model, key);
      const errorFeatName =
          this.getColumnName(model, key, CalculatedColumnType.ERROR);
      const sqErrorFeatName =
          this.getColumnName(model, key, CalculatedColumnType.SQUARED_ERROR);
      const scores = regressionResults.map(
          (result: RegressionResults) => result[key].score);
      const errors = regressionResults.map(
          (result: RegressionResults) => result[key].error);
      const sqErrors = regressionResults.map(
          (result: RegressionResults) => result[key].squared_error);
      const dataType = createLitType(Scalar);
      const source = `${REGRESSION_SOURCE_PREFIX}:${model}`;
      this.addColumnFromList(
          scores, data, key, scoreFeatName, dataType, source);
      if (output[key] instanceof RegressionScore) {
        const predSpec = output[key] as RegressionScore;
        if (predSpec.parent != null) {
          this.addColumnFromList(
              errors, data, key, errorFeatName, dataType, source);
          this.addColumnFromList(
              sqErrors, data, key, sqErrorFeatName, dataType, source);
        }
      }
    }
  }

  /**
   * Run scalar predictions and store results in data service.
   */
  private async runScalarPreds(model: string, data: IndexedInput[]) {
    const {output} = this.appState.currentModelSpecs[model].spec;
    if (findSpecKeys(output, Scalar).length === 0) {
      return;
    }

    const predsPromise = this.apiService.getPreds(
        data, model, this.appState.currentDataset, [Scalar]);
    const preds = await predsPromise;

    // Add scalar results as new column to the data service.
    if (preds == null || preds.length === 0) {
      return;
    }
    const scalarKeys = Object.keys(preds[0]);
    for (const key of scalarKeys) {
      const scoreFeatName = this.getColumnName(model, key);
      const scores = preds.map(pred => pred[key]);
      const dataType = createLitType(Scalar);
      const source = `${SCALAR_SOURCE_PREFIX}:${model}`;
      this.addColumnFromList(
          scores, data, key, scoreFeatName, dataType, source);
    }
  }

  @action
  async setValuesForNewDatapoints(datapoints: IndexedInput[]) {
    // When new datapoints are created, set their data values for each
    // column stored in the data service.
    for (const input of datapoints) {
      for (const col of this.cols) {
        const key = col.name;
        const val = await this.columnHeaders.get(key)!.getValueFn(input);
        this.columnData.get(key)!.set(input.id, val);
      }
    }
  }

  @computed
  get cols(): DataColumnHeader[] {
    return Array.from(this.columnHeaders.values());
  }

  getColNamesOfType(litTypeType: typeof LitType): string[] {
    return this.cols.filter(col => isLitSubtype(col.dataType, litTypeType))
                    .map(col => col.name);
  }

  getColumnInfo(name: string): DataColumnHeader|undefined {
    return this.columnHeaders.get(name);
  }

  /** Flattened list of values in data columns for reacting to data changes. **/
  // TODO(b/156100081): Can we get observers to react to changes to columnData
  // without needing this computed list?
  @computed
  get dataVals() {
    const vals: ValueType[] = [];
    for (const colVals of this.columnData.values()) {
      vals.push(...colVals.values());
    }
    return vals;
  }

  /**
   * Add new column to data service, including values for existing datapoints.
   *
   * If column has been previously added, replaces the existing data with new
   * data, if they are different.
   *
   * @param {ColumnData} columnVals Map from datapoint ID to values to associate
   *     with this column.
   * @param {string} key the field name associated with this column in the
   *     Dataset's or Model's spec.
   * @param {string} name the display name for this column. For Dataset (i.e.,
   *     input) columns these are typically the same as `key`. For columns
   *     asociated with a Model's output spec, use DataService.getColumnName()
   *     to generate the name to use here.
   * @param {LitType} dataType the LitType (i.e., the type used in Specs)
   *     corresonding to the values in this column.
   * @param {Source} source a colon-separated (:) string describing the origins
   *     of the values in this column. These are used by modules -- e.g.,
   *     Scalars -- to determine which columns to draw from for display. For
   *     model predictions, source should always contain the model name,
   *     optionally preceeded by one of the *_SOURCE_PREFIX constants exported
   *     by this module.
   * @param {ValueFn} getValueFn function for getting new values for this
   *     column, typically called when a user adds a new Datapoint to the
   *     DataService through the Datapoint Editor or a Generator.
   * @param {ColorRange=} colorRange a color range to associate with values from
   *     this column.
   */
  @action
  addColumn(
      columnVals: ColumnData, key: string, name: string, dataType: LitType,
      source: Source, getValueFn: ValueFn = () => null,
      colorRange?: ColorRange) {
    if (!this.columnHeaders.has(name)) {
      this.columnHeaders.set(
          name, {dataType, source, name, key, getValueFn, colorRange});
    }
    // TODO(b/156100081): If data service table is properly observable, may
    // be able to get rid of this check.
    if (!this.columnData.has(name) ||
        !mapsContainSame(this.columnData.get(name)!, columnVals)) {
      this.columnData.set(name, columnVals);
    }
  }

  /**
   * Add new column to data service, including values for existing datapoints.
   *
   * If column has been previously added, replaces the existing data with new
   * data, if they are different.
   */
  @action
  addColumnFromList(
      values: ValueType[], data: IndexedInput[], key: string, name: string,
      dataType: LitType, source: Source, getValueFn: ValueFn = () => null,
      colorRange?: ColorRange) {
    if (values.length !== data.length) {
      throw new Error(`Attempted to add data column ${
          name} with incorrect number of values.`);
    }
    const columnVals: ColumnData = new Map();
    for (let i = 0; i < data.length; i++) {
      const input = data[i];
      const value = values[i];
      columnVals.set(input.id, value);
    }
    this.addColumn(
        columnVals, key, name, dataType, source, getValueFn, colorRange);
  }

  /** Get stored value for a datapoint ID for the provided column key. */
  getVal(id: string, key: string) {
    // If column not tracked by data service, try to get the value from the
    // input data through appState.
    if (!this.columnHeaders.has(key)) {
      const inputData = this.appState.getCurrentInputDataById(id);
      return inputData != null ? inputData.data[key] : null;
    }
    // If no value yet stored for this datapoint for this column, return null.
    if (!this.columnData.get(key)!.has(id)) {
      return null;
    }
    return this.columnData.get(key)!.get(id);
  }

  /**
   * Asynchronously get value for a datapoint ID for the provided column key.
   *
   *  This method is async as if the value has not yet been been retrieved
   *  for a new datapoint, it will return the promise fetching the value.
   */
  async getValAsync(id: string, key: string) {
    if (!this.columnHeaders.has(key) || this.columnData.get(key)!.has(id)) {
      return this.getVal(id, key);
    }

    const input = this.appState.getCurrentInputDataById(id)!;
    const val = await this.columnHeaders.get(key)!.getValueFn(input);
    this.columnData.get(key)!.set(input.id, val);
    return val;
  }

  /** Get list of column values from all datapoints. */
  getColumn(key: string): ValueType[] {
    // Map from the current input data, as opposed to getting from the data
    // service's columnData as the columnData might have some missing entries
    // for new datapoints where the value hasn't been asyncronously-returned.
    // This way, we ensure we get a list of values, one per datapoint, with
    // nulls for datapoints with no info for that column in the data service
    // yet.
    return this.appState.currentInputData.map(
        input => this.getVal(input.id, key));
  }
}
