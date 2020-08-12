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

import {autorun} from 'mobx';
import {LitService} from './lit_service';

/**
 * Interface for reading/storing app configuration from/to the URL.
 */
export class UrlConfiguration {
  selectedTab?: string;
  selectedModels: string[] = [];
  selectedData: string[] = [];
  primarySelectedData?: string;
  selectedDataset?: string;
  hiddenModules: string[] = [];
  compareExamplesEnabled?: boolean;
  layout?: string;
}

/**
 * Interface describing how AppState is synced to the URL Service
 */
export interface StateObservedByUrlService {
  currentModels: string[];
  currentDataset: string;
  setUrlConfiguration: (urlConfiguration: UrlConfiguration) => void;
  compareExamplesEnabled: boolean;
  layoutName: string;
}

/**
 * Interface describing how the ModulesService is synced to the URL Service
 */
export interface ModulesObservedByUrlService {
  hiddenModuleKeys: Set<string>;
  setUrlConfiguration: (urlConfiguration: UrlConfiguration) => void;
  selectedTab: string;
}

/**
 * Interface describing how the SelectionService is synced to the URL service
 */
export interface SelectionObservedByUrlService {
  readonly primarySelectedId: string|null;
  setPrimarySelection: (id: string) => void;
  readonly selectedIds: string[];
  selectIds: (ids: string[]) => void;
}

const SELECTED_TAB_KEY = 'tab';
const SELECTED_DATA_KEY = 'selection';
const PRIMARY_SELECTED_DATA_KEY = 'primary';
const SELECTED_DATASET_KEY = 'dataset';
const SELECTED_MODELS_KEY = 'models';
const HIDDEN_MODULES_KEY = 'hidden_modules';
const COMPARE_EXAMPLES_ENABLED_KEY = 'compare_data_mode';
const LAYOUT_KEY = 'layout';

const MAX_IDS_IN_URL_SELECTION = 100;

/**
 * Singleton service responsible for deserializing / serializing state to / from
 * a url.
 */
export class UrlService extends LitService {
  /** Parse arrays in a url param, filtering out empty strings */
  private urlParseArray(encoded: string) {
    const array = encoded.split(',');
    return array.filter(str => str !== '');
  }

  /** Parse a string in a url param, filtering out empty strings */
  private urlParseString(encoded: string) {
    return encoded ? encoded : undefined;
  }

  /** Parse a boolean in a url param, if undefined return false */
  private urlParseBoolean(encoded: string) {
    return encoded === 'true';
  }

  private getConfigurationFromUrl(): UrlConfiguration {
    const urlConfiguration = new UrlConfiguration();

    const urlSearchParams = new URLSearchParams(window.location.search);
    urlSearchParams.forEach((value: string, key: string) => {
      if (key === SELECTED_MODELS_KEY) {
        urlConfiguration.selectedModels = this.urlParseArray(value);
      } else if (key === SELECTED_DATA_KEY) {
        urlConfiguration.selectedData = this.urlParseArray(value);
      } else if (key === PRIMARY_SELECTED_DATA_KEY) {
        urlConfiguration.primarySelectedData = this.urlParseString(value);
      } else if (key === SELECTED_DATASET_KEY) {
        urlConfiguration.selectedDataset = this.urlParseString(value);
      } else if (key === HIDDEN_MODULES_KEY) {
        urlConfiguration.hiddenModules = this.urlParseArray(value);
      } else if (key === COMPARE_EXAMPLES_ENABLED_KEY) {
        urlConfiguration.compareExamplesEnabled = this.urlParseBoolean(value);
      } else if (key === SELECTED_TAB_KEY) {
        urlConfiguration.selectedTab = this.urlParseString(value);
      } else if (key === LAYOUT_KEY) {
        urlConfiguration.layout = this.urlParseString(value);
      }
    });

    return urlConfiguration;
  }

  /** Set url parameter if it's not empty */
  private setUrlParam(
      params: URLSearchParams, key: string, data: string|string[]) {
    const value = data instanceof Array ? data.join(',') : data;
    if (value !== '') {
      params.set(key, value);
    }
  }

  /**
   * Parse the URL configuration and set it in the services that depend on it
   * for initializtion. Then, set up an autorun observer to automatically
   * react to changes of state and sync them to the url query params.
   */
  syncStateToUrl(
      appState: StateObservedByUrlService,
      selectionService: SelectionObservedByUrlService,
      modulesService: ModulesObservedByUrlService) {
    const urlConfiguration = this.getConfigurationFromUrl();
    appState.setUrlConfiguration(urlConfiguration);
    modulesService.setUrlConfiguration(urlConfiguration);

    const urlSelectedIds = urlConfiguration.selectedData || [];
    selectionService.selectIds(urlSelectedIds);
    if (urlConfiguration.primarySelectedData != null) {
      selectionService.setPrimarySelection(
          urlConfiguration.primarySelectedData);
    }
    // TODO(lit-dev) Add compared examples to URL parameters.
    // Only enable compare example mode if both selections and compare mode
    // exist in URL.
    if (selectionService.selectedIds.length > 0 &&
        urlConfiguration.compareExamplesEnabled) {
      appState.compareExamplesEnabled = true;
    }

    autorun(() => {
      const urlParams = new URLSearchParams();

      // Syncing app state
      this.setUrlParam(urlParams, SELECTED_MODELS_KEY, appState.currentModels);
      if (selectionService.selectedIds.length <= MAX_IDS_IN_URL_SELECTION) {
        this.setUrlParam(
            urlParams, SELECTED_DATA_KEY, selectionService.selectedIds);
        if (selectionService.primarySelectedId != null) {
          this.setUrlParam(
              urlParams, PRIMARY_SELECTED_DATA_KEY,
              selectionService.primarySelectedId);
        }
      }
      this.setUrlParam(
          urlParams, SELECTED_DATASET_KEY, appState.currentDataset);

      this.setUrlParam(
          urlParams, COMPARE_EXAMPLES_ENABLED_KEY,
          appState.compareExamplesEnabled ? 'true' : 'false');

      // Syncing hidden modules
      this.setUrlParam(
          urlParams, HIDDEN_MODULES_KEY, [...modulesService.hiddenModuleKeys]);

      this.setUrlParam(urlParams, SELECTED_TAB_KEY, modulesService.selectedTab);

      this.setUrlParam(urlParams, LAYOUT_KEY, appState.layoutName);

      if (urlParams.toString() !== '') {
        const newUrl = `${window.location.pathname}?${urlParams.toString()}`;
        window.history.replaceState({}, '', newUrl);
      }
    });
  }
}
