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

/**
 * Testing for the color service.
 */

import 'jasmine';

import {LitApp} from '../core/app';
import {IndexedInput} from '../lib/types';
import {DEFAULT, CATEGORICAL_NORMAL} from '../lib/colors';
import {mockMetadata} from '../lib/testing_utils';

import {ColorService, SignedSalienceCmap, UnsignedSalienceCmap} from './color_service';
import {DataService} from './data_service';
import {GroupService} from './group_service';
import {AppState} from './state_service';

describe('Color service test', () => {
  const categoricalFeatures = {
    'testFeat0': ['0', '1'],
    'testFeat1': ['a', 'b', 'c']
  };
  const numericalFeatureRanges = {
    'testNumFeat0': [-5, 5],
    'testNumFeat1': [0, 1],
  };
  const booleanFeatureNames: string[] = ['testBool'];
  const categoricalFeatureNames = Object.keys(categoricalFeatures);
  const numericalFeatureNames = Object.keys(numericalFeatureRanges);
  // It seems you can't mock mobx @computed values since they're read-only, so
  // this is a hack to get around that.
  const mockGroupService = {
    categoricalFeatureNames,
    numericalFeatureNames,
    booleanFeatureNames,
    categoricalFeatures,
    numericalFeatureRanges
  } as unknown as GroupService;


  const app = new LitApp();
  const colorService = new ColorService(
      mockGroupService, app.getService(DataService));

  it('Tests colorableOption', () => {
    const opts = colorService.colorableOptions;

    // Default value + two cat features + 2 numerical features +
    // 1 bool feature = 6 options total.
    expect(opts.length).toBe(6);

    // Test a categorical option.
    const colorOpt = opts[0];
    expect(colorOpt.name).toEqual('testFeat0');
    expect(colorOpt.getValue).toBeInstanceOf(Function);
    expect(colorOpt.scale).toBeInstanceOf(Function);

    // Test a categorical option.
    const colorOpt1 = opts[2];
    expect(colorOpt1.name).toEqual('testNumFeat0');
    expect(colorOpt1.getValue).toBeInstanceOf(Function);
    expect(colorOpt1.scale).toBeInstanceOf(Function);
  });

  it('Tests getDatapointColor(), selectedColorOption, and reset()', () => {
    const dataMap = new Map<string, IndexedInput>();
    const mockInput: IndexedInput = {
      id: 'xxxxxxx',
      data: {'testFeat0': 1, 'testNumFeat0': 0},
      meta: {}
    };
    dataMap.set('xxxxxxx', mockInput);

    const inputData = new Map<string, Map<string, IndexedInput>>();
    inputData.set('color_test', dataMap);

    const appState = app.getService(AppState);
    // Stop appState from trying to make the call to the back end
    // to load the data (causes test flakiness.)
    spyOn(appState, 'loadData').and.returnValue(Promise.resolve());
    // tslint:disable-next-line:no-any (to spyOn a private, readonly property)
    spyOnProperty<any>(appState, 'inputData', 'get').and.returnValue(inputData);
    appState.metadata = mockMetadata;
    appState.setCurrentDataset('color_test');

    // With no change in settings, the color should be the default color.
    let color = colorService.getDatapointColor(mockInput);
    expect(color).toEqual(DEFAULT);

    // When the settings are updated, getDatapointColor() should reflect the
    // update.
    colorService.selectedColorOptionName =
        colorService.colorableOptions[0].name;
    color = colorService.getDatapointColor(mockInput);
    expect(color).toEqual(CATEGORICAL_NORMAL[1]);

    // Updating to a numerical color scheme.
    colorService.selectedColorOptionName =
        colorService.colorableOptions[2].name;
    color = colorService.getDatapointColor(mockInput);
    expect(color).toEqual('rgb(51, 138, 163)');
  });

  it('provides color map classes for salience viz', () => {
    const signedCmap = new SignedSalienceCmap();
    const unsignedCmap = new UnsignedSalienceCmap();

    expect(signedCmap.lightness(0)).toEqual(0);
    expect(signedCmap.lightness(0.5)).toEqual(0.5);
    expect(signedCmap.lightness(1)).toEqual(1);

    expect(unsignedCmap.lightness(0)).toEqual(0);
    expect(unsignedCmap.lightness(0.5)).toEqual(0.5);
    expect(unsignedCmap.lightness(1)).toEqual(1);

    expect(signedCmap.bgCmap(-1)).toEqual('rgb(71, 0, 70)');
    expect(signedCmap.bgCmap(0)).toEqual('rgb(255, 254, 254)');
    expect(signedCmap.bgCmap(1)).toEqual('rgb(4, 30, 53)');

    expect(unsignedCmap.bgCmap(0)).toEqual('rgb(255, 255, 255)');
    expect(unsignedCmap.bgCmap(1)).toEqual('rgb(40, 1, 135)');

    expect(signedCmap.textCmap(-1)).toEqual('white');
    expect(signedCmap.textCmap(0)).toEqual('black');
    expect(signedCmap.textCmap(1)).toEqual('white');

    expect(unsignedCmap.textCmap(0)).toEqual('black');
    expect(unsignedCmap.textCmap(1)).toEqual('white');
  });
});
