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
import {autorun, toJS} from 'mobx';

import {Constructor} from '../lib/types';
import {ApiService} from '../services/api_service';
import {ClassificationService} from '../services/classification_service';
import {ColorService} from '../services/color_service';
import {DataService} from '../services/data_service';
import {FocusService} from '../services/focus_service';
import {GroupService} from '../services/group_service';
import {LitService} from '../services/lit_service';
import {ModulesService} from '../services/modules_service';
import {SelectionService} from '../services/selection_service';
import {SettingsService} from '../services/settings_service';
import {SliceService} from '../services/slice_service';
import {AppState} from '../services/state_service';
import {StatusService} from '../services/status_service';
import {UrlService} from '../services/url_service';


/**
 * The class responsible for building and managing the LIT App.
 */
export class LitApp {
  constructor() {
    this.buildServices();
  }

  /**
   * Begins loading data from the LIT server, and computes the layout that
   * the `modules` component will use to render.
   */
  async initialize() {
    const apiService = this.getService(ApiService);
    const appState = this.getService(AppState);
    const modulesService = this.getService(ModulesService);
    const [selectionService, pinnedSelectionService] =
        this.getServiceArray(SelectionService);
    const urlService = this.getService(UrlService);
    const colorService = this.getService(ColorService);

    // Load the app metadata before any further initialization
    appState.metadata = await apiService.getInfo();
    console.log('[LIT - metadata]', toJS(appState.metadata));

    // Update page title based on metadata
    if (appState.metadata.pageTitle) {
      document.querySelector('html head title')!.textContent =
          appState.metadata.pageTitle;
    }

    // Sync app state based on URL search params
    urlService.syncStateToUrl(appState, modulesService, selectionService,
        pinnedSelectionService, colorService);

    // Initialize the rest of the app state
    await appState.initialize();

    // Initilize the module layout
    modulesService.initializeLayout(
        appState.layout, appState.currentModelSpecs,
        appState.currentDatasetSpec, appState.compareExamplesEnabled);

    // Select the initial datapoint, if one was set in the URL.
    await urlService.syncSelectedDatapointToUrl(appState, selectionService);

    // Enabling comparison mode if a datapoint has been pinned
    if (pinnedSelectionService.primarySelectedId) {
      appState.compareExamplesEnabled = true;
    }

    // If enabled, set up state syncing back to Python.
    if (appState.metadata.syncState) {
      autorun(() => {
        apiService.pushState(
            selectionService.selectedInputData, appState.currentDataset, {
              'primary_id': selectionService.primarySelectedId,
              'pinned_id': pinnedSelectionService.primarySelectedId,
            });
      });
    }
  }

  private readonly services =
      new Map<Constructor<LitService>, LitService|LitService[]>();

  /** Simple DI service system */
  getService<T extends LitService>(t: Constructor<T>, instance?: string): T {
    let service = this.services.get(t);
    /**
     * Modules that don't support example comparison will always get index
     * 0 of selectionService. This way we do not have to edit any module that
     * does not explicitly support cloning. For modules that support comparison,
     * if the `pinned` instance is specified then return the appropriate
     * instance.
     */
    if (Array.isArray(service)) {
      if (instance != null && instance !== 'pinned') {
        throw new Error(`Invalid service instance name: ${instance}`);
      }
      service = service[instance === 'pinned' ? 1 : 0];
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
    const urlService = new UrlService(apiService);
    const appState = new AppState(apiService, statusService);
    const selectionService = new SelectionService(appState);
    const pinnedSelectionService = new SelectionService(appState);
    const sliceService = new SliceService(selectionService, appState);
    const settingsService =
        new SettingsService(appState, modulesService, selectionService);
    const classificationService = new ClassificationService(appState);
    const dataService = new DataService(
        appState, classificationService, apiService, settingsService);
    const groupService = new GroupService(appState, dataService);
    const colorService = new ColorService(groupService, dataService);
    const focusService = new FocusService(selectionService);

    // Populate the internal services map for dependency injection
    this.services.set(ApiService, apiService);
    this.services.set(AppState, appState);
    this.services.set(ClassificationService, classificationService);
    this.services.set(ColorService, colorService);
    this.services.set(DataService, dataService);
    this.services.set(FocusService, focusService);
    this.services.set(GroupService, groupService);
    this.services.set(ModulesService, modulesService);
    this.services.set(SelectionService, [
      selectionService, pinnedSelectionService
    ]);
    this.services.set(SettingsService, settingsService);
    this.services.set(SliceService, sliceService);
    this.services.set(StatusService, statusService);
    this.services.set(UrlService, urlService);
  }
}

/** The exported singleton instance of the LIT App */
export const app = new LitApp();
