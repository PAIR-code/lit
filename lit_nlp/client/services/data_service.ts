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
import {IndexedInput, LitType} from '../lib/types';

import {LitService} from './lit_service';
import {AppState} from './state_service';


/** Data source for a data column. */
export type Source = string;

/** Info about a data column. */
export interface DataCol {
  dataType: LitType;
  name: string;
  source: Source;
}

/** Data value and info for a value in a data column. */
export interface DataValue {
  dataType: LitType;
  //tslint:disable-next-line:no-any
  value: any;
  source: Source;
}

/** Map of data column names to values for a datapoint. */
export type DataForInput = Map<string, DataValue>;

/**
 * Data service singleton, responsible for maintaining a table of keys and
 * values for each data point.
 */
export class DataService extends LitService {
  @observable data: DataForInput[] = [];

  constructor(private readonly appState: AppState) {
    super();
    reaction(() => appState.currentDataset, currentDataset => {
      this.data = [];
    });
    reaction(() => appState.currentInputData, currentInputData => {
      this.setTable(currentInputData);
    });
  }

  @computed
  get cols(): DataCol[] {
    if (this.data.length === 0) {
      return [];
    }
    const keys = Array.from(this.data[0].keys());
    return keys.map(key => {
      return {name: key, dataType: this.data[0].get(key)!.dataType,
              source: this.data[0].get(key)!.source};
    });
  }

  @computed
  get scalarCols(): string[] {
    return this.cols.filter(col => col.dataType.__name__ === 'Scalar').map(
        col => col.name);
  }


  setTable(inputData: IndexedInput[]) {
    const data: DataForInput[] = [];
    const spec = this.appState.currentDatasetSpec;
    const keys = Object.keys(spec);

    for (const input of inputData) {
      const dataForInput: DataForInput = new Map();
      for (const key of keys) {
        dataForInput.set(key, {
          dataType: spec[key],
          source: 'Data',
          value: input.data[key]
        });
      }
      data.push(dataForInput);
    }
    this.data = data;
  }

  @action //tslint:disable-next-line:no-any
  addColumn(columnData: any[], key: string, dataType: LitType, source: Source) {
    for (let i = 0; i < columnData.length; i++) {
      const dataForInput = columnData[i];
      const point = this.data[i];
      point.set(key, {
        dataType,
        source,
        value: dataForInput
      });
    }
    this.data = [...this.data];
  }
}

