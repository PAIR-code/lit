/**
 * @fileoverview Tests for validating the behavior of the GroupService
 *
 * @license
 * Copyright 2021 Google LLC
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
import {DataService} from './data_service';
import {GroupService, FacetingConfig, FacetingMethod} from './group_service';
import {AppState} from './state_service';
import {Scalar} from '../lib/lit_types';
import {mockMetadata} from '../lib/testing_utils';
import {IndexedInput} from '../lib/types';
import {findSpecKeys} from '../lib/utils';

describe('GroupService test', () => {
  const booleanFeature: FacetingConfig = {
    featureName: 'isAlive'
  };
  const categoricalFeature: FacetingConfig = {
    featureName: 'species'
  };
  const discreteConfig: FacetingConfig = {
    featureName: 'culmen_depth_mm',
    method: FacetingMethod.DISCRETE
  };
  const equalIntervalConfig: FacetingConfig = {
    featureName: 'culmen_length_mm',
    method: FacetingMethod.EQUAL_INTERVAL,
    numBins: 3
  };
  const quantileConfig: FacetingConfig = {
    featureName: 'flipper_length_mm',
    method: FacetingMethod.QUANTILE,
    numBins: 4
  };
  const thresholdConfig: FacetingConfig = {
    featureName: 'body_mass_g',
    method: FacetingMethod.THRESHOLD,
    threshold: 4500
  };

  const validConfigs = [
    booleanFeature,
    categoricalFeature,
    discreteConfig,
    equalIntervalConfig,
    quantileConfig,
    thresholdConfig
  ];

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

  let appState: AppState, groupService: GroupService, dataService: DataService;

  beforeEach(async () => {
    // Set up.
    const app = new LitApp();
    const inputData = new Map<string, Map<string, IndexedInput>>();
    inputData.set('penguin_dev', penguinData);

    appState = app.getService(AppState);
    dataService = app.getService(DataService);
    // Stop appState from trying to make the call to the back end
    // to load the data (causes test flakiness.)
    spyOn(appState, 'loadData').and.returnValue(Promise.resolve());
    appState.metadata = mockMetadata;
    // tslint:disable-next-line:no-any (to spyOn a private, readonly property)
    spyOnProperty<any>(appState, 'inputData', 'get').and.returnValue(inputData);
    appState.setCurrentDataset('penguin_dev');

    groupService = new GroupService(appState, dataService);
  });

  it('validates FacetingConfig', () => {
    const badQuantThresh: FacetingConfig = {
      featureName: 'body_mass_g',
      method: FacetingMethod.QUANTILE,
      threshold: 0.5
    };

    const badThreshNumBins: FacetingConfig = {
      featureName: 'body_mass_g',
      method: FacetingMethod.THRESHOLD,
      numBins: 10
    };

    expect(() => { groupService.validateFacetingConfigs([badQuantThresh]); })
      .toThrow(new Error(`Invalid faceting config for 'body_mass_g': ` +
                         `{"featureName":"body_mass_g","method":"quantile",` +
                         `"threshold":0.5}`));

    expect(() => { groupService.validateFacetingConfigs([badThreshNumBins]); })
      .toThrow(new Error(`Invalid faceting config for 'body_mass_g': ` +
                         `{"featureName":"body_mass_g","method":"threshold",` +
                         `"numBins":10}`));
  });

  it('converts valid configs into a string sorted by feature name', () => {
    const key1 = groupService.facetConfigsToKey(validConfigs);
    const key2 = groupService.facetConfigsToKey([
      equalIntervalConfig,
      quantileConfig,
      thresholdConfig,
      booleanFeature,
      categoricalFeature,
      discreteConfig,
    ]);

    expect(key1).toEqual(key2);
  });

  it('calculates ranges for all numerical features', () => {
    const dataSpec = appState.currentDatasetSpec;
    const names = findSpecKeys(dataSpec, Scalar);

    for (const name of names) {
      const range = groupService.numericalFeatureRanges[name];
      expect(range).toBeInstanceOf(Array);
      expect(range.length).toEqual(2);
      expect(typeof range[0]).toEqual('number');
      expect(typeof range[1]).toEqual('number');
    }
  });

  it('generates bins for numerical features', () => {
    const discreteBins = groupService.numericalFeatureBins([discreteConfig]);
    expect(Object.keys(discreteBins[discreteConfig.featureName]).length)
        .toEqual(90);

    const eqIntInfBins = groupService.numericalFeatureBins([{
      ...equalIntervalConfig,
      numBins: -1   // Triggers bin generation with Freedman-Diaconis algorithm
    }]);
    expect(Object.keys(eqIntInfBins[equalIntervalConfig.featureName]).length)
        .toEqual(3);

    const eqIntExpBins =
        groupService.numericalFeatureBins([equalIntervalConfig]);
    expect(Object.keys(eqIntExpBins[equalIntervalConfig.featureName]).length)
        .toEqual(equalIntervalConfig.numBins as number);

    const quantBins = groupService.numericalFeatureBins([quantileConfig]);
    expect(Object.keys(quantBins[quantileConfig.featureName]).length)
        .toEqual(4);

    const theshBins = groupService.numericalFeatureBins([thresholdConfig]);
    expect(Object.keys(theshBins[thresholdConfig.featureName]).length)
        .toEqual(2);
  });

  it('generates a label for bin combinations', () => {
    const features = validConfigs.map(c => c.featureName);
    const bins = groupService.numericalFeatureBins(validConfigs);
    const label = groupService.numIntersectionsLabel(bins, features);
    expect(label).toEqual('2x90x3x4x2x3 = 12960');
  });

  it('groups the mock penguin data into the correct numeric bins', () => {
    const data = [...penguinData.values()];
    const numericConfigs = [
      discreteConfig,
      equalIntervalConfig,
      quantileConfig,
      thresholdConfig
    ];

    for (const config of numericConfigs) {
      // Skip discrete binning in this test as the number of bins might not
      // match the grouping as bins without datapoints aren't returned by
      // the groupExamplesByFeatures method.
      if (config.method === 'discrete') {
        continue;
      }
      const feature = config.featureName;
      const bins = groupService.numericalFeatureBins([config]);
      const facets = Object.keys(bins[feature]);
      const grps = groupService.groupExamplesByFeatures(bins, data, [feature]);
      expect(Object.keys(grps).length).toEqual(facets.length);
    }
  });
});
