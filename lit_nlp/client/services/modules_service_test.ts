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

import {toJS} from 'mobx';

import {LitApp} from '../core/app';
import {mockMetadata} from '../lib/testing_utils';
import {LitCanonicalLayout} from '../lib/types';
import {AttentionModule} from '../modules/attention_module';
import {DatapointEditorModule} from '../modules/datapoint_editor_module';

import {ApiService} from './api_service';
import {ModulesService} from './modules_service';
import {AppState} from './state_service';

const MOCK_LAYOUT: LitCanonicalLayout = {
  upper: {
    'Main': [
      'datapoint-editor-module',
    ],
  },
  lower: {
    'internals': [
      // Duplicated per model and in compareDatapoints mode.
      'attention-module',
    ],
  },
  layoutSettings: {hideToolbar: true, mainHeight: 90, centerPage: true},
  description: 'Mock layout for testing.'
};

describe('modules service test', () => {
  let appState: AppState, modulesService: ModulesService;
  beforeEach(async () => {
    // Set up.
    const app = new LitApp();
    // tslint:disable-next-line:no-any (to spyOn a private method)
    spyOn<any>(app.getService(ApiService), 'queryServer').and.returnValue(null);
    appState = app.getService(AppState);
    // Stop appState from trying to make the call to the back end
    // to load the data (causes test flakiness.)
    spyOn(appState, 'loadData').and.returnValue(Promise.resolve());

    appState.metadata = mockMetadata;
    await appState.setCurrentModels(['sst_0_micro']);
    appState.setCurrentDataset('sst_dev');
    // Stop all calls to the backend (causes test flakiness.)
    modulesService = app.getService(ModulesService);
  });

  it('tests setHiddenModules', () => {
    const hiddenModuleKeys = ['hiddenKey0', 'hiddenKey1'];
    modulesService.setHiddenModules(hiddenModuleKeys);
    const updatedModuleKeys = new Set(toJS(modulesService.hiddenModuleKeys));
    expect(updatedModuleKeys).toEqual(new Set(hiddenModuleKeys));
  });

  it('tests initializeLayout', () => {
    modulesService.initializeLayout(
        MOCK_LAYOUT, appState.currentModelSpecs, appState.currentDatasetSpec,
        false);
    expect(modulesService.declaredLayout).toEqual(MOCK_LAYOUT);
  });

  it('tests updateRenderLayout (standard)', () => {
    modulesService.declaredLayout = MOCK_LAYOUT;
    const compareExamples = false;
    modulesService.updateRenderLayout(
        appState.currentModelSpecs, appState.currentDatasetSpec,
        compareExamples);

    // Check that the component groups are the same.
    const updatedLayout = modulesService.getRenderLayout();
    expect(Object.keys(updatedLayout.upper))
        .toEqual(Object.keys(MOCK_LAYOUT.upper));
    expect(Object.keys(updatedLayout.lower))
        .toEqual(Object.keys(MOCK_LAYOUT.lower));

    // Check that the two modules we added to the layout are reflected in
    // allModuleKeys.
    const keys =
        new Set([`Main_${DatapointEditorModule.title}`,
                 `internals_${AttentionModule.title}`]);
    expect(modulesService.allModuleKeys).toEqual(keys);
  });

  it('tests updateRenderLayout (comparing examples)', () => {
    modulesService.declaredLayout = MOCK_LAYOUT;
    const compareExamples = true;
    modulesService.updateRenderLayout(
        appState.currentModelSpecs, appState.currentDatasetSpec,
        compareExamples);

    // Check that the render configs duplicated correctly for the modules
    // that should be duplicated when comparing examples.
    const configs = modulesService.getRenderLayout().lower['internals'];
    expect(configs[0].length).toEqual(2);  // Two examples to compare.
  });

  it('tests updateRenderLayout (comparing models)', async () => {
    modulesService.declaredLayout = MOCK_LAYOUT;
    await appState.setCurrentModels(['sst_0_micro', 'sst_1_micro']);
    const compareExamples = false;
    modulesService.updateRenderLayout(
        appState.currentModelSpecs, appState.currentDatasetSpec,
        compareExamples);

    // Check that the render configs duplicated correctly for the modules
    // that should be duplicated when comparing models.
    const configs = modulesService.getRenderLayout().lower['internals'];
    expect(configs[0].length).toEqual(2);  // Two models to compare.
  });

  it('tests updateRenderLayout (comparing models and examples)', async () => {
    modulesService.declaredLayout = MOCK_LAYOUT;
    await appState.setCurrentModels(['sst_0_micro', 'sst_1_micro']);
    const compareExamples = true;
    modulesService.updateRenderLayout(
        appState.currentModelSpecs, appState.currentDatasetSpec,
        compareExamples);

    // Check that the render configs duplicated correctly for the modules
    // that should be duplicated when comparing models.
    const configs = modulesService.getRenderLayout().lower['internals'];
    expect(configs[0].length).toEqual(4);  // Two models x two examples = 4.
  });
});
