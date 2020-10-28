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

// Import Services
// Import and add injection functionality to LitModule
import {reaction} from 'mobx';

import {Constructor} from '../lib/types';

import {ApiService} from '../services/api_service';
import {ClassificationService} from '../services/classification_service';
import {ColorService} from '../services/color_service';
import {GroupService} from '../services/group_service';
import {LitService} from '../services/lit_service';
import {ModulesService, LitComponentLayouts} from '../services/modules_service';
import {RegressionService} from '../services/regression_service';
import {SelectionService} from '../services/selection_service';
import {SettingsService} from '../services/settings_service';
import {SliceService} from '../services/slice_service';
import {PredictionsService} from '../services/predictions_service';
import {AppState} from '../services/state_service';
import {StatusService} from '../services/status_service';
import {UrlService} from '../services/url_service';
import {DeltasService} from '../services/deltas_service';


/**
 * The class responsible for building and managing the LIT App.
 */
export class LITApp {
  constructor() {
    this.buildServices();
  }

  /**
   * Begins loading data from the LIT server, and computes the layout that
   * the `modules` component will use to render.
   */
  async initialize(layouts: LitComponentLayouts) {
    const appState = this.getService(AppState);
    const modulesService = this.getService(ModulesService);

    await appState.initialize();
    const layout = layouts[appState.layoutName];
    modulesService.initializeLayout(
        layout, appState.currentModelSpecs, appState.currentDatasetSpec,
        appState.compareExamplesEnabled);

    /**
     * If we need more than one selectionService, create and append it to the
     * list.
     */
    const numSelectionServices = modulesService.numberOfSelectionServices;
    const selectionServices = this.getServiceArray(SelectionService);
    for (let i = 0; i < numSelectionServices - 1; i++) {
      const selectionService = new SelectionService();
      selectionService.setAppState(appState);
      selectionServices.push(selectionService);
    }
    /**
     * Reaction to sync other selection services to selections of the main one.
     */
    reaction(() => appState.compareExamplesEnabled, compareExamplesEnabled => {
      this.syncSelectionServices();
    }, {fireImmediately: true});
  }

  private readonly services =
      new Map<Constructor<LitService>, LitService|LitService[]>();

  /** Sync selection services */
  syncSelectionServices() {
    const selectionServices = this.getServiceArray(SelectionService);
    selectionServices.slice(1).forEach((selectionService: SelectionService) => {
      // TODO(lit-dev): can we just copy the object instead, and skip this
      // logic?
      selectionService.syncFrom(selectionServices[0]);
    });
  }

  /** Simple DI service system */
  getService<T extends LitService>(t: Constructor<T>): T {
    let service = this.services.get(t);
    /**
     * Modules that don't support example comparison will always get index
     * 0 of selectionService. This way we do not have to edit any module that
     * does not explicitly support cloning
     */
    if (Array.isArray(service)) {
      service = service[0];
    }
    if (service === undefined) {
      throw new Error(`Service is undefined: ${t.name}`);
    }
    return service as T;
  }

  /**
   * Intended for selectionService only, returns an array of services for
   * indexing within modules.
   */
  getServiceArray<T extends LitService>(t: Constructor<T>): T[] {
    const services = this.services.get(t) as T[];
    if (services === undefined) {
      throw new Error(`Service is undefined: ${t.name}`);
    }
    return services;
  }

  /**
   * Builds services via simple constructor / composition based dependency
   * injection. We'll might want to come up with something more robust down the
   * line, but for now this allows us to construct all of our singleton
   * services in one location in a simple way.
   */
  private buildServices() {
    const statusService = new StatusService();
    const apiService = new ApiService(statusService);
    const modulesService = new ModulesService();
    const selectionService = new SelectionService();
    const urlService = new UrlService();
    const appState = new AppState(apiService, statusService);
    const sliceService = new SliceService(selectionService, appState);
    const classificationService =
        new ClassificationService(apiService, appState);
    const regressionService = new RegressionService(apiService, appState);
    const settingsService =
        new SettingsService(appState, modulesService, selectionService);
    const groupService = new GroupService(appState);
    const colorService = new ColorService(
        appState, groupService, classificationService, regressionService);
    const deltasService = new DeltasService(
        appState, selectionService, classificationService, regressionService);
    const predictionsService = new PredictionsService(appState, apiService,
        classificationService, regressionService);
    selectionService.setAppState(appState);

    // Initialize url syncing of state
    urlService.syncStateToUrl(appState, selectionService, modulesService);

    // Populate the internal services map for dependency injection
    this.services.set(ApiService, apiService);
    this.services.set(AppState, appState);
    this.services.set(ClassificationService, classificationService);
    this.services.set(ColorService, colorService);
    this.services.set(GroupService, groupService);
    this.services.set(ModulesService, modulesService);
    this.services.set(RegressionService, regressionService);
    this.services.set(SelectionService, [selectionService]);
    this.services.set(SettingsService, settingsService);
    this.services.set(SliceService, sliceService);
    this.services.set(StatusService, statusService);
    this.services.set(UrlService, urlService);
    this.services.set(DeltasService, deltasService);
    this.services.set(PredictionsService, predictionsService);
  }
}

/** The exported singleton instance of the LIT App */
export const app = new LITApp();
