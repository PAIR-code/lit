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
import {action, computed, reaction} from 'mobx';

import {arrayContainsSame} from '../lib/utils';

import {LitService} from './lit_service';
import {AppState, ModulesService, SelectionService} from './services';

/**
 * Parameters used to update the global settings, and compute if a rerender
 * needs to happen.
 */
export interface UpdateSettingsParams {
  models?: string[];
  dataset?: string;
  layoutName?: string;
}

/**
 * A service to manage global app settings updates and configuration.
 */
export class SettingsService extends LitService {
  constructor(
      private readonly appState: AppState,
      private readonly modulesService: ModulesService,
      private readonly selectionService: SelectionService) {
    super();
    // If compare examples changes, update layout using the 'quick' path.
    reaction(() => appState.compareExamplesEnabled, compareExamplesEnabled => {
      this.modulesService.quickUpdateLayout(
          this.appState.currentModelSpecs, this.appState.currentDatasetSpec,
          compareExamplesEnabled);
    });
  }

  isDatasetValidForModels(dataset: string, models: string[]) {
    return models.every(model => this.isDatasetValidForModel(dataset, model));
  }

  private isDatasetValidForModel(dataset: string, model: string) {
    const availableDatasets =
        this.appState.metadata?.models?.[model].datasets || [];
    return availableDatasets.includes(dataset);
  }

  @computed
  get isValidCurrentDataAndModels() {
    return this.isDatasetValidForModels(
        this.appState.currentDataset, this.appState.currentModels);
  }

  /**
   * Update settings and do a 'full refresh' of all modules.
   * Use this if changing models or datasets to ensure that modules don't make
   * API calls with inconsistent state while the update is in progress.
   */
  @action
  async updateSettings(updateParams: UpdateSettingsParams) {
    // Clear all modules.
    this.modulesService.clearLayout();

    // After one animation frame, the modules have been cleared. Now make
    // settings changes and recompute the layout.
    await this.raf();
    const nextModels = updateParams.models ?? this.appState.currentModels;
    const nextDataset = updateParams.dataset ?? this.appState.currentDataset;
    const nextLayout = updateParams.layoutName ?? this.appState.layoutName;

    // Compare the updated models
    const haveModelsChanged =
        !arrayContainsSame(this.appState.currentModels, nextModels);

    const hasDatasetChanged =
        nextDataset !== this.appState.currentDataset && nextDataset;

    if (haveModelsChanged) {
      await this.appState.setCurrentModels(nextModels);
    }

    // Compare the updated dataset
    if (hasDatasetChanged) {
      // Unselect all selected points if the dataset is changing.
      this.selectionService.selectIds([]);

      this.appState.setCurrentDataset(
          nextDataset, /** load data */
          true);
    }

    // If the entire layout has changed, reinitialize the layout.
    if (this.appState.layoutName !== nextLayout) {
      this.appState.layoutName = nextLayout;
      this.modulesService.initializeLayout(
        this.appState.layout, this.appState.currentModelSpecs,
        this.appState.currentDatasetSpec, this.appState.compareExamplesEnabled);
      this.modulesService.renderModules();
    } else {
      // Recompute layout using the 'quick' path.
      this.modulesService.quickUpdateLayout(
          this.appState.currentModelSpecs, this.appState.currentDatasetSpec,
          this.appState.compareExamplesEnabled);
    }
  }

  private async raf() {
    return new Promise(resolve => {
      requestAnimationFrame(resolve);
    });
  }
}
