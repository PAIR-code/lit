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

import * as d3 from 'd3';

import {LitApp} from '../core/app';
import {IndexedInput} from '../lib/types';

import {ClassificationService} from './classification_service';
import {ColorService} from './color_service';
import {GroupService} from './group_service';
import {RegressionService} from './regression_service';
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
  const categoricalFeatureNames = Object.keys(categoricalFeatures);
  const numericalFeatureNames = Object.keys(numericalFeatureRanges);
  // It seems you can't mock mobx @computed values since they're read-only, so
  // this is a hack to get around that.
  const mockGroupService = {
    categoricalFeatureNames,
    numericalFeatureNames,
    categoricalFeatures,
    numericalFeatureRanges
  } as unknown as GroupService;


  const app = new LitApp();
  const colorService = new ColorService(
      app.getService(AppState), mockGroupService,
      app.getService(ClassificationService), app.getService(RegressionService));

  it('Tests colorableOption', () => {
    const opts = colorService.colorableOptions;

    // Default value + two cat features + 2 numerical features = 3 options
    // total.
    expect(opts.length).toBe(5);

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
    const mockInput: IndexedInput = {
      id: 'xxxxxxx',
      data: {'testFeat0': 1, 'testNumFeat0': 0},
      meta: {}
    };

    // With no change in settings, the color should be the default color.
    let color = colorService.getDatapointColor(mockInput);
    expect(color).toEqual(d3.schemeCategory10[0]);

    // When the settings are updated, getDatapointColor() should reflect the
    // update.
    colorService.selectedColorOption = colorService.colorableOptions[0];
    color = colorService.getDatapointColor(mockInput);
    expect(color).toEqual(d3.schemeCategory10[1]);

    // Updating to a numerical color scheme.
    colorService.selectedColorOption = colorService.colorableOptions[2];
    color = colorService.getDatapointColor(mockInput);
    expect(color).toEqual('#21918c');

    // After resetting, getDatapointColor() should reset.
    colorService.reset();
    color = colorService.getDatapointColor(mockInput);
    expect(color).toEqual(d3.schemeCategory10[0]);
  });
});
