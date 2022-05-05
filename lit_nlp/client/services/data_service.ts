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

import {IndexedInput, LitName, LitType} from '../lib/types';
import {isLitSubtype} from '../lib/utils';

import {LitService} from './lit_service';
import {AppState} from './state_service';


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
  source: Source;
  getValueFn: ValueFn;
}

/** Map of datapoint ID to values for a column of data. */
export type ColumnData = Map<string, ValueType>;

/**
 * Data service singleton, responsible for maintaining columns of computed data
 * for datapoints in the current dataset.
 */
export class DataService extends LitService {
  @observable private readonly columnHeaders =
      new Map<string, DataColumnHeader>();
  @observable readonly columnData = new Map<string, ColumnData>();

  constructor(
      private readonly appState: AppState) {
    super();
    reaction(() => appState.currentDataset, () => {
      this.columnHeaders.clear();
      this.columnData.clear();
    });

    this.appState.addNewDatapointsCallback(async (newDatapoints) =>
      this.setValuesForNewDatapoints(newDatapoints));
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

  getColNamesOfType(typeName: LitName): string[] {
    return this.cols.filter(col => isLitSubtype(col.dataType, typeName)).map(
        col => col.name);
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
   */
  @action
  addColumn(
      columnVals: ColumnData, name: string, dataType: LitType, source: Source,
      getValueFn: ValueFn = () => null) {
    if (!this.columnHeaders.has(name)) {
      this.columnHeaders.set(name, {dataType, source, name, getValueFn});
    }
    if (!this.columnData.has(name) || (
            JSON.stringify(columnVals.values()) !==
            JSON.stringify(this.columnData.get(name)!.values()))) {
      this.columnData.set(name, columnVals);
    }
  }

  /** Get stored value for a datapoint ID for the provided column key. */
  getVal(id: string, key: string) {
    // If column not tracked by data service, get value from input data through
    // appState.
    if (!this.columnHeaders.has(key)) {
      return this.appState.getCurrentInputDataById(id)!.data[key];
    }
    // If no value yet stored for this datapoint for this column, return null.
    if (!this.columnData.get(key)!.has(id)) {
      return null;
    }
    return this.columnData.get(key)!.get(id);
  }

  /** Asyncronously get value for a datapoint ID for the provided column key.
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
