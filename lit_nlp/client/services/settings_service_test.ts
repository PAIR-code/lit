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

import 'jasmine';

import {LitApp} from '../core/app';
import {mockMetadata} from '../lib/testing_utils';
import {IndexedInput} from '../lib/types';

import {SettingsService} from './settings_service';
import {AppState} from './state_service';


describe('isDatasetValid test', () => {
  // Set up.
  const app = new LitApp();
  app.getService(AppState).metadata = mockMetadata;
  const settingsService = app.getService(SettingsService);

  it('tests dataset validity when correct', () => {
    const dataset = 'sst_dev';
    const models = ['sst_0_micro', 'sst_0_micro'];
    const isValid = settingsService.isDatasetValidForModels(dataset, models);
    expect(isValid).toBe(true);
  });

  it('tests dataset validity when incorrect', () => {
    const dataset = 'non_matching_dataset';
    const models = ['sst_0_micro', 'sst_0_micro'];
    const isValid = settingsService.isDatasetValidForModels(dataset, models);
    expect(isValid).toBe(false);
  });
});

describe('updateSettings test', () => {
  it('sets correct settings', async () => {
    // Set up services
    const app = new LitApp();
    const appState = app.getService(AppState);
    // Use mock data.
    appState.metadata = mockMetadata;
    appState.addLayouts(appState.metadata.layouts);
    appState.layoutName = appState.metadata.defaultLayout;
    appState.setDatasetForTest('sst_dev', new Map<string, IndexedInput>());
    appState.setDatasetForTest('dummy', new Map<string, IndexedInput>());
    const settingsService = app.getService(SettingsService);

    // Update the data
    const hiddenModuleKeys = new Set(['moduleKey0', 'moduleKey1']);
    const models = ['sst_0_micro'];
    const dataset = 'sst_dev';
    const updateParams = {models, dataset, hiddenModuleKeys};
    await settingsService.updateSettings(updateParams);

    // Check the currentDataset and currentModels values directly.
    expect(appState.currentDataset).toEqual(dataset);
    expect(appState.currentModels).toEqual(models);
  });
});
