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
import {action, observable} from 'mobx';

import {LayoutSettings, type LitCanonicalLayout, LitComponentSpecifier, LitModuleClass, LitModuleConfig, LitTabGroupLayout, type ModelInfoMap, ResolvedModuleConfig, type Spec} from '../lib/types';

import {LitService} from './lit_service';
import {ModulesObservedByUrlService, UrlConfiguration} from './url_service';

/**
 * An interface describing how to render a LIT module, specifying which module
 * to render and whether it renders on a per-model basis.
 */
export interface RenderConfig {
  key: string;
  moduleType: LitModuleClass;
  modelName?: string;
  selectionServiceIndex?: number;
}

/**
 * Layout for a single group of tabs, each containing multiple modules,
 * optionally replicated.
 */
export interface LitTabGroupConfig {
  [tabName: string]: RenderConfig[][];
}

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
  upper: LitTabGroupConfig;
  lower: LitTabGroupConfig;
  left: LitTabGroupConfig;
}

type RenderModulesCallback = () => void;

/**
 * Look up any module names given as strings, and return the
 * constructor object.
 */
function getModuleConstructor(moduleName: string): LitModuleClass {
  const moduleClass = window.customElements.get(moduleName);
  if (moduleClass === undefined) {
    throw new Error(
        `Malformed layout; unable to find element '${moduleName}'`);
  }
  return moduleClass as unknown as LitModuleClass;
}

/**
 * Create a canonical module config from a specifier, which may just be a
 * string.
 * TODO(lit-dev): run this when canonicalizing layout?
 */
export function resolveModuleConfig(
    specifier: LitComponentSpecifier): ResolvedModuleConfig {
  const moduleConfig: LitModuleConfig =
      typeof specifier === 'string' ? {module: specifier} : specifier;

  const constructor = getModuleConstructor(moduleConfig.module);
  return Object.assign(
      {
        constructor,
        title: constructor.title,
        requiredForTab: moduleConfig.requiredForTab ?? false
      },
      moduleConfig);
}

/**
 * Singleton service responsible for maintaining which modules to render.
 */
export class ModulesService extends LitService implements
    ModulesObservedByUrlService {
  @observable declaredLayout: LitCanonicalLayout = {
    upper: {}, lower: {}, left: {}, layoutSettings: {}, description: ''
  };
  @observable readonly selectedTabs = {upper: '', lower: '', left: ''};
  @observable private renderLayout: LitRenderConfig = {
    upper: {}, lower: {}, left: {}
  };
  @observable hiddenModuleKeys = new Set<string>();
  @observable expandedModuleKey = '';
  allModuleKeys = new Set<string>();
  private renderModulesCallback: RenderModulesCallback = () => {};

  // TODO(b/168201937): Remove imperative logic and use observables/reactions
  // for all module logic.

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
      layout: LitCanonicalLayout, currentModelInfos: ModelInfoMap,
      datasetSpec: Spec, compareExamples: boolean) {
    this.declaredLayout = layout;
    this.updateRenderLayout(currentModelInfos, datasetSpec, compareExamples);
  }

  @action
  clearLayout() {
    this.updateRenderLayout({}, {}, /* compareExamples */ false);
    this.renderModules();
  }

  /**
   * Recompute layout without clearing modules.
   * Use this for comparison mode to provide a faster refresh, and avoid
   * clearing module-level state such as datatable filters.
   */
  @action
  quickUpdateLayout(
      currentModelInfos: ModelInfoMap, datasetSpec: Spec,
      compareExamples: boolean) {
    // Recompute layout
    this.updateRenderLayout(currentModelInfos, datasetSpec, compareExamples);
    this.renderModules();
  }

  setHiddenModules(keys: Set<string>|string[]) {
    // Ensure we copy to a new set
    const nextHiddenModuleKeys = new Set<string>(keys);
    this.hiddenModuleKeys = nextHiddenModuleKeys;
  }

  setExpandedModule(key: string) {
    this.expandedModuleKey = key;
  }

  setUrlConfiguration(urlConfiguration: UrlConfiguration) {
    this.setHiddenModules(urlConfiguration.hiddenModules);
    this.setExpandedModule(urlConfiguration.expandedModule ?? '');
    this.selectedTabs.upper = urlConfiguration.selectedTabUpper ?? '';
    this.selectedTabs.lower = urlConfiguration.selectedTabLower ?? '';
    this.selectedTabs.left = urlConfiguration.selectedTabLeft ?? '';
  }

  isModuleGroupHidden(config: RenderConfig) {
    return this.hiddenModuleKeys.has(config.key);
  }

  isModuleGroupExpanded(config: RenderConfig) {
    return this.expandedModuleKey === config.key;
  }

  toggleHiddenModule(config: RenderConfig, isHidden: boolean) {
    if (isHidden) {
      this.hiddenModuleKeys.add(config.key);
    } else {
      this.hiddenModuleKeys.delete(config.key);
    }
  }

  toggleExpandedModule(config: RenderConfig, isExpanded: boolean) {
    if (isExpanded) {
      this.expandedModuleKey = config.key;
    } else {
      this.expandedModuleKey = '';
    }
  }

  getRenderLayout(): LitRenderConfig {
    return this.renderLayout;
  }

  getSetting(settingName: keyof LayoutSettings) {
    // settingName is guaranteed to be a keyof layoutSettings.
    // tslint:disable-next-line:no-dict-access-on-struct-type
    return this.declaredLayout?.layoutSettings?.[settingName];
  }

  /**
   * Compute module configurations to render determining whether or not a module
   * is visible for the selected models and user settings, and whether to render
   * copies of a module per model based on the module behavior.
   */
  computeTabGroupLayout(
      groupContents: LitTabGroupLayout, currentModelInfos: ModelInfoMap,
      datasetSpec: Spec, compareExamples: boolean) {
    const tabGroupConfig: LitTabGroupConfig = {};
    for (const [tabName, tabContents] of Object.entries(groupContents)) {
      const moduleConfigs = tabContents.map(resolveModuleConfig);

      // First, map all of the modules to render configs, filtering out those
      // that are not visible.
      const configs = this.getRenderConfigs(
          moduleConfigs, currentModelInfos, datasetSpec, compareExamples,
          tabName);

      if (configs.length !== 0) {
        tabGroupConfig[tabName] = configs;
      }
    }
    return tabGroupConfig;
  }

  updateRenderLayout(
      currentModelInfos: ModelInfoMap, datasetSpec: Spec,
      compareExamples: boolean) {
    const upper = this.computeTabGroupLayout(
        this.declaredLayout.upper, currentModelInfos, datasetSpec,
        compareExamples);
    const lower = this.computeTabGroupLayout(
        this.declaredLayout.lower, currentModelInfos, datasetSpec,
        compareExamples);
    const left = this.computeTabGroupLayout(
        this.declaredLayout.left, currentModelInfos, datasetSpec,
        compareExamples);

    const renderLayout: LitRenderConfig = {upper, lower, left};

    this.updateModuleKeys(renderLayout);
    this.renderLayout = renderLayout;
  }

  private updateModuleKeys(renderLayout: LitRenderConfig) {
    const allModuleKeys = new Set<string>();
    for (const section of Object.values(renderLayout) as LitTabGroupConfig[]) {
      for (const tabGroup of Object.values(section)) {
        for (const configGroup of tabGroup) {
          for (const config of configGroup) {
            allModuleKeys.add(config.key);
            if (config.moduleType.collapseByDefault) {
              this.hiddenModuleKeys.add(config.key);
            }
          }
        }
      }
    }
    // Clean up any extraneous hidden module keys that are not part of the
    // allModuleKeys set
    for (const key of [...this.hiddenModuleKeys]) {
      if (!allModuleKeys.has(key)) {
        this.hiddenModuleKeys.delete(key);
      }
    }

    this.allModuleKeys = allModuleKeys;
  }

  /**
   * Process module configs for a single tab.
   *
   * Generates module renderConfig object or objects for a given LIT module
   * depending on the model specs. Since some modules can render one copy per
   * model, this method specifies the configurations for those multiple modules
   * to render.
   */
  private getRenderConfigs(
      modules: ResolvedModuleConfig[], currentModelInfos: ModelInfoMap,
      datasetSpec: Spec, compareExamples: boolean,
      tabName: string): RenderConfig[][] {
    const renderConfigs: RenderConfig[][] = [];
    // Iterate over all modules to generate render config objects, expanding
    // modules that display one per model.
    for (const moduleConfig of modules) {
      const moduleType = moduleConfig.constructor;
      if (!moduleType.shouldDisplayModule(currentModelInfos, datasetSpec)) {
        if (moduleConfig.requiredForTab) {
          // Abort this tab if a required module is not compatible.
          return [];
        } else {
          // Otherwise, skip this module and continue with others.
          continue;
        }
      }

      const configs: RenderConfig[] = [];

      // Is compare examples mode on and does this module require duplication?
      const compare =
          compareExamples && moduleType.duplicateForExampleComparison;
      const key = `${tabName}_${moduleType.title}`;

      // model = undefined means the resulting module(s) will not be keyed to
      // a specific model; they can access the list of active models via
      // this.appState.currentModels.
      let selectedModels: Array<string|undefined> = [undefined];
      if (moduleType.duplicateForModelComparison) {
        selectedModels = Object.keys(currentModelInfos);
      }
      for (const modelName of selectedModels) {
        if (compare) {
          // The 'reference' selection service is index 1, but we want this to
          // render on top/left, so create this config first.
          configs.push(this.makeRenderConfig(key, moduleType, modelName, 1));
          configs.push(this.makeRenderConfig(key, moduleType, modelName, 0));
        } else {
          configs.push(this.makeRenderConfig(
              key, moduleType, modelName, compareExamples ? 0 : undefined));
        }
      }
      // Don't push empty configs, e.g. if there is no model selected.
      if (configs.length > 0) {
        renderConfigs.push(configs);
      } else {
        console.warn(`LIT layout: no compatible configs for module '${
            moduleType.title}' in tab '${tabName}'`);
      }
    }

    return renderConfigs;
  }

  private makeRenderConfig(
      key: string, moduleType: LitModuleClass, modelName?: string,
      selectionServiceIndex?: number): RenderConfig {
    return {
      key,
      moduleType,
      modelName,
      selectionServiceIndex,
    };
  }
}
