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
import {observable} from 'mobx';

import {LitModuleClass, ModelInfoMap, Spec} from '../lib/types';
import {LayoutSettings, LitComponentLayout} from '../lib/types';

import {LitService} from './lit_service';
import {ModulesObservedByUrlService, UrlConfiguration} from './url_service';

/**
 * A layout is defined by a set of main components that are always visible,
 * (designated in the object by the "main" key)
 * and a set of tabs that each contain a group other components.
 *
 * LitRenderConfig is a mapping of tab names arrays of component lists.
 * Each component list includes multiple instances of the same component,
 * duplicated for each model. This basically corresponds to the grid of
 * components that will be rendered.
 */
export interface LitRenderConfig {
  [name: string]: RenderConfig[][];
}

/**
 * An interface describing how to render a LIT module, specifying which module
 * to render and whether it renders on a per-model basis.
 */
export interface RenderConfig {
  moduleType: LitModuleClass;
  tab: string;
  modelName?: string;
  selectionServiceIndex?: number;
}

type RenderModulesCallback = () => void;

/**
 * Singleton service responsible for maintaining which modules to render.
 */
export class ModulesService extends LitService implements
    ModulesObservedByUrlService {
  @observable
  declaredLayout: LitComponentLayout = {'components': {}, 'layoutSettings': {}};
  @observable selectedTab: string = '';
  private renderLayout: LitRenderConfig = {};
  private renderModulesCallback: RenderModulesCallback = () => {};

  /**
   * We need to make the rendering of modules an explicit, callback-driven
   * update because of difficulty ensuring the template results of the
   * lit-modules don't trigger detach/reattach behavior of the module
   * components.
   */
  setRenderModulesCallback(callback: RenderModulesCallback) {
    this.renderModulesCallback = callback;
  }

  /**
   * Explicitly calls the render modules callback, which is set to rerender
   * the lit-modules component.
   */
  renderModules() {
    this.renderModulesCallback();
  }

  /**
   * In app initialization, we need to set the declared layout and compute the
   * visible render layout based on the app config
   */
  initializeLayout(
      layout: LitComponentLayout, currentModelSpecs: ModelInfoMap,
      datasetSpec: Spec, compareExamples: boolean) {
    this.setModuleLayout(layout);
    this.declaredLayout.layoutSettings = layout.layoutSettings || {};
    this.computeRenderLayout(currentModelSpecs, datasetSpec, compareExamples);
  }
  @observable hiddenModuleKeys = new Set<string>();
  /**
   * This module calculates how many selectionServices will be required based
   * on the module layout. If the app enters compare example mode ModulesService
   * checks every modules behavior to determine if it will be cloned or not,
   * sets up the new layout with cloned modules and updates this to required
   * number of selectionServices. Currently this number is fixed at 2.
   */
  @observable numberOfSelectionServices: number = 2;

  allModuleKeys = new Set<string>();

  setHiddenModules(keys: Set<string>|string[]) {
    // Ensure we copy to a new set
    const nextHiddenModuleKeys = new Set<string>(keys);
    this.hiddenModuleKeys = nextHiddenModuleKeys;
  }

  setUrlConfiguration(urlConfiguration: UrlConfiguration) {
    this.setHiddenModules(urlConfiguration.hiddenModules);
    this.selectedTab = urlConfiguration.selectedTab ?? '';
  }

  isModuleGroupHidden(config: RenderConfig) {
    const key = this.getModuleKey(config);
    return this.hiddenModuleKeys.has(key);
  }

  toggleHiddenModule(config: RenderConfig, isHidden: boolean) {
    const key = this.getModuleKey(config);
    if (isHidden) {
      this.hiddenModuleKeys.add(key);
    } else {
      this.hiddenModuleKeys.delete(key);
    }
  }

  getRenderLayout() {
    return this.renderLayout;
  }

  getSetting(settingName: keyof LayoutSettings) {
    return this.declaredLayout?.layoutSettings?.[settingName];
  }

  /**
   * Compute module configurations to render determining whether or not a module
   * is visible for the selected models and user settings, and whether to render
   * copies of a module per model based on the module behavior.
   */
  computeRenderLayout(
      currentModelSpecs: ModelInfoMap, datasetSpec: Spec,
      compareExamples: boolean) {
    const renderLayout: LitRenderConfig = {};
    const allModuleKeys = new Set<string>();

    const componentGroupNames = Object.keys(this.declaredLayout.components);
    componentGroupNames.forEach(groupName => {
      const components = this.declaredLayout.components[groupName];
      // First, map all of the modules to render configs, filtering out those
      // that are not visible.
      const configs = this.getRenderConfigs(
          components, currentModelSpecs, datasetSpec, compareExamples, groupName);
      configs.forEach(configGroup => {
        configGroup.forEach(config => {
          const key = this.getModuleKey(config);
          allModuleKeys.add(key);

          if (config.moduleType.collapseByDefault) {
            this.hiddenModuleKeys.add(key);
          }
        });
      });
      renderLayout[groupName] = configs;
    });

    this.allModuleKeys = allModuleKeys;

    // Clean up any extraneous hidden module keys that are not part of the
    // allModuleKeys set
    for (const key of [...this.hiddenModuleKeys]) {
      if (!this.allModuleKeys.has(key)) {
        this.hiddenModuleKeys.delete(key);
      }
    }
    this.renderLayout = renderLayout;
  }

  /**
   * Generates module renderConfig object or objects for a given LIT module
   * depending on the model specs. Since some modules can render one copy per
   * model, this method specifies the configurations for those multiple modules
   * to render.
   */
  private getRenderConfigs(
      modules: LitModuleClass[], currentModelSpecs: ModelInfoMap,
      datasetSpec: Spec, compareExamples: boolean, tab: string) {
    const renderConfigs: RenderConfig[][] = [];
    // Iterate over all modules to generate render config objects, expanding
    // modules that display one per model.
    modules.forEach(moduleType => {
      // Is compare examples mode on and does this module require duplication?
      const compare =
          compareExamples && moduleType.duplicateForExampleComparison;
      if (!moduleType.shouldDisplayModule(currentModelSpecs, datasetSpec)) {
        return;
      } else if (!moduleType.duplicateForModelComparison) {
        const config: RenderConfig[] = [];
        if (compare) {
          config.push(this.makeRenderConfig(moduleType, tab, undefined, 1));
          config.push(this.makeRenderConfig(moduleType, tab, undefined, 0));
        } else {
          config.push(this.makeRenderConfig(
              moduleType, tab, undefined, compareExamples ? 0 : undefined));
        }
        renderConfigs.push(config);
      } else {
        const selectedModels = Object.keys(currentModelSpecs);
        const configs =
            selectedModels.reduce((accArray: RenderConfig[], modelName) => {
              if (compare) {
                accArray.push(this.makeRenderConfig(moduleType, tab, modelName, 1));
                accArray.push(this.makeRenderConfig(moduleType, tab, modelName, 0));
              } else {
                accArray.push(this.makeRenderConfig(
                    moduleType, tab, modelName, compareExamples ? 0 : undefined));
              }
              return accArray;
            }, []);
        renderConfigs.push(configs);
      }
    });

    return renderConfigs;
  }

  private makeRenderConfig(
      moduleType: LitModuleClass, tab: string, modelName?: string,
      selectionServiceIndex?: number): RenderConfig {
    return {
      moduleType,
      tab,
      modelName,
      selectionServiceIndex,
    };
  }

  private getModuleKey(config: RenderConfig) {
    return `${config.tab}_${config.moduleType.title}`;
  }

  setModuleLayout(layout: LitComponentLayout) {
    this.declaredLayout = layout;
  }
}
