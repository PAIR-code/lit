/**
 * @fileoverview Tests for validating the behavior of the DataService
 *
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

import 'jasmine';

import {LitApp} from '../core/app';
import {Scalar, TextSegment} from '../lib/lit_types';
import {mockMetadata} from '../lib/testing_utils';
import {IndexedInput} from '../lib/types';
import {createLitType} from '../lib/utils';
import {ApiService, AppState, ClassificationService, SettingsService, StatusService} from '../services/services';
import {ColumnData, DataService} from './data_service';

describe('DataService test', () => {
  const penguinData = new Map<string, IndexedInput>();
  penguinData.set('a', { id:'a', meta:{}, data:{
      body_mass_g: 3098,
      culmen_depth_mm: 22,
      culmen_length_mm: 45,
      flipper_length_mm: 172,
      isAlive: true}});
  penguinData.set('b', { id:'b', meta:{}, data:{
      body_mass_g: 3559,
      culmen_depth_mm: 21,
      culmen_length_mm: 32,
      flipper_length_mm: 224,
      isAlive: true}});
  penguinData.set('c', { id:'c', meta:{}, data:{
      body_mass_g: 3217,
      culmen_depth_mm: 19,
      culmen_length_mm: 53,
      flipper_length_mm: 214,
      isAlive: true}});
  penguinData.set('d', { id:'d', meta:{}, data:{
      body_mass_g: 2700,
      culmen_depth_mm: 17,
      culmen_length_mm: 34,
      flipper_length_mm: 172,
      isAlive: true}});
  penguinData.set('e', { id:'e', meta:{}, data:{
      body_mass_g: 4301,
      culmen_depth_mm: 14,
      culmen_length_mm: 48,
      flipper_length_mm: 206,
      isAlive: true}});
  penguinData.set('f', { id:'f', meta:{}, data:{
      body_mass_g: 5081,
      culmen_depth_mm: 15,
      culmen_length_mm: 60,
      flipper_length_mm: 203,
      isAlive: true}});
  penguinData.set('g', { id:'g', meta:{}, data:{
      body_mass_g: 3085,
      culmen_depth_mm: 13,
      culmen_length_mm: 46,
      flipper_length_mm: 177,
      isAlive: true}});
  penguinData.set('h', { id:'h', meta:{}, data:{
      body_mass_g: 4584,
      culmen_depth_mm: 16,
      culmen_length_mm: 57,
      flipper_length_mm: 188,
      isAlive: true}});
  penguinData.set('i', { id:'i', meta:{}, data:{
      body_mass_g: 2804,
      culmen_depth_mm: 18,
      culmen_length_mm: 59,
      flipper_length_mm: 221,
      isAlive: true}});
  penguinData.set('j', { id:'j', meta:{}, data:{
      body_mass_g: 3725,
      culmen_depth_mm: 15,
      culmen_length_mm: 47,
      flipper_length_mm: 211,
      isAlive: true}});
  penguinData.set('k', { id:'k', meta:{}, data:{
      body_mass_g: 3615,
      culmen_depth_mm: 13,
      culmen_length_mm: 36,
      flipper_length_mm: 213,
      isAlive: true}});
  penguinData.set('l', { id:'l', meta:{}, data:{
      body_mass_g: 6300,
      culmen_depth_mm: 17,
      culmen_length_mm: 43,
      flipper_length_mm: 231,
      isAlive: true}});

  let appState: AppState;
  let classificationService: ClassificationService;
  let dataService: DataService;
  const apiService = new ApiService(new StatusService());
  let settingsService: SettingsService;

  beforeEach(async () => {
    // Set up.
    const app = new LitApp();
    const inputData = new Map<string, Map<string, IndexedInput>>();
    inputData.set('penguin_dev', penguinData);
    appState = app.getService(AppState);
    // Stop appState from trying to make the call to the back end
    // to load the data (causes test flakiness.)
    spyOn(appState, 'loadData').and.returnValue(Promise.resolve());
    appState.metadata = mockMetadata;
    // tslint:disable-next-line:no-any (to spyOn a private, readonly property)
    spyOnProperty<any>(appState, 'inputData', 'get').and.returnValue(inputData);
    appState.setCurrentDataset('penguin_dev');

    settingsService = app.getService(SettingsService);
    classificationService = app.getService(ClassificationService);
    dataService = new DataService(
        appState, classificationService, apiService, settingsService);
  });

  it('has correct columns', () => {
    expect(dataService.cols.length).toBe(0);

    const dataType = createLitType(Scalar);
    const getValueFn = () => 1;
    const dataMap: ColumnData = new Map();
    for (let i = 0; i < appState.currentInputData.length; i++) {
      const input = appState.currentInputData[i];
      dataMap.set(input.id, 1);
    }
    dataService.addColumn(
        dataMap, 'featKey', 'newFeat', dataType, 'Test', getValueFn);

    expect(dataService.cols.length).toBe(1);
    expect(dataService.cols[0].key).toBe('featKey');
    expect(dataService.cols[0].name).toBe('newFeat');
    expect(dataService.getColNamesOfType(Scalar).length).toBe(1);
    expect(dataService.getColNamesOfType(TextSegment).length).toBe(0);
  });

  it('has correct column data', () => {
    const dataType = createLitType(Scalar);
    const getValueFn = () => 1;
    const dataMap: ColumnData = new Map();
    for (let i = 0; i < appState.currentInputData.length; i++) {
      const input = appState.currentInputData[i];
      dataMap.set(input.id, i);
    }
    dataService.addColumn(
        dataMap, 'featKey', 'newFeat', dataType, 'Test', getValueFn);

    expect(dataService.getVal('a', 'newFeat')).toBe(0);
    expect(dataService.getVal('b', 'newFeat')).toBe(1);
    expect(dataService.getVal('notReal', 'newFeat')).toBeNull();

    const colVals = dataService.getColumn('newFeat');
    expect(colVals).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  });

  it('handles new datapoints', async () => {
    const dataType = createLitType(Scalar);
    const getValueFn = () => 1000;
    const dataMap: ColumnData = new Map();
    for (let i = 0; i < appState.currentInputData.length; i++) {
      const input = appState.currentInputData[i];
      dataMap.set(input.id, i);
    }
    dataService.addColumn(
        dataMap, 'featKey', 'newFeat', dataType, 'Test', getValueFn);

    expect(dataService.getVal('newDatapoint', 'newFeat')).toBeNull();
    const newDatapoint = {
      id:'newDatapoint',
      meta:{},
      data:{
        body_mass_g: 3725,
        culmen_depth_mm: 15,
        culmen_length_mm: 47,
        flipper_length_mm: 211,
        isAlive: true
      }
    };
    await dataService.setValuesForNewDatapoints([newDatapoint]);

    expect(dataService.getVal('newDatapoint', 'newFeat')).toBe(1000);
    expect(await dataService.getValAsync('newDatapoint', 'newFeat')).toBe(1000);
  });

  it('handles pass-through to input data', () => {
    expect(dataService.cols.length).toBe(0);

    expect(dataService.getVal('a', 'body_mass_g')).toBe(3098);
    expect(dataService.getVal('a', 'fake_feature')).toBeUndefined();
  });
});
