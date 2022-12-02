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
import {html, TemplateResult} from 'lit';
import {property} from 'lit/decorators';
import {computed, observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {LitModuleClass, ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {ApiService, AppState, SelectionService} from '../services/services';

import {app} from './app';

/**
 * An interface describing the LitWidget element that contains the LitModule.
 */
export interface ParentWidgetElement {
  isLoading: boolean;
}

type IsLoadingFn = (isLoading: boolean) => void;
type OnScrollFn = () => void;

/**
 * The base class from which all Lit Module classes extends, in order to have
 * type safety for dynamically creating modules. Derives from MobxLitElement for
 * automatic reactive rendering. Provides a few helper methods for setting up
 * explicit mobx reactions with automatic disposal upon component disconnect.
 */
export abstract class LitModule extends ReactiveElement {
  /**
   * A callback used to set the loading status of the parent widget component.
   */
  @property({type: Object}) setIsLoading: IsLoadingFn = (status: boolean) => {};

  /**
   * A callback used to keep scrolling syncronized between duplicated instances
   * of a module. Only used if the class defined by SCROLL_SYNC_CSS_CLASS is
   * used in an element in the module. Otherwise scrolling is syncronized using
   * the outer container that contains the module.
   */
  @observable @property({type: Object}) onSyncScroll: OnScrollFn|null = null;

  // Name of this module, to show in the UI.
  static title: string = '';

  // A URL to the reference documentation.
  static referenceURL: string = '';

  // Number of columns of the 12 column horizontal layout.
  static numCols: number = 4;

  // Whether to collapse this module by default.
  static collapseByDefault: boolean = false;

  // If true, duplicate this module in example comparison mode.
  static duplicateForExampleComparison: boolean = false;

  // If true, duplicate this module when running with more than one model.
  static duplicateForModelComparison: boolean = true;

  // If true, duplicate this module as rows, instead of columns.
  static duplicateAsRow: boolean = false;

  // Template function. Should return HTML to create this element in the DOM.
  static template:
      (model: string, selectionServiceIndex: number,
       shouldReact: number) => TemplateResult = () => html``;

  @property({type: String}) model = '';
  @observable @property({type: Number}) selectionServiceIndex = 0;

  // tslint:disable-next-line:no-any
  protected readonly latestLoadPromises = new Map<string, Promise<any>>();

  protected readonly apiService = app.getService(ApiService);
  protected readonly appState = app.getService(AppState);

  @computed
  protected get selectionService() {
    return app.getServiceArray(SelectionService)[this.selectionServiceIndex];
  }

  override updated() {
    // If the class defined by SCROLL_SYNC_CSS_CLASS is used in the module then
    // set its onscroll callback to be the provided onSyncScroll.
    // There is no need to use this class if a module scrolls through the
    // normal mechanism of its parent container div from the LitWidget element
    // that wraps modules. But if a module doesn't scroll using that parent
    // container, but through some element internal to the module, then using
    // this class on that element will allow for scrolling to be syncronized
    // across duplicated modules of this type.
    const scrollElems = this.shadowRoot!.querySelectorAll(
        `.${SCROLL_SYNC_CSS_CLASS}`);
    scrollElems.forEach(elem => {
      (elem as HTMLElement).onscroll = this.onSyncScroll;
    });
  }

  /**
   * Base module render function - not to be overridden by clients.
   *
   * This render function will call the renderImpl method if the module is set
   * to react, otherwise it will not render anything.
   *
   * Any client overridding this method will not get the standard behavior of
   * pausing rendering when a module is set to not react. This may cause issues
   * as the module will still have reactions paused in this case. Therefore,
   * clients should avoid overidding this method and instead they should
   * implement renderImpl.
   */
  override render() {
    // If the module is not reactive, then do not render anything.
    if (this.shouldReact === 0) {
      return;
    }
    return this.renderImpl();
  }

  /**
   * Render function for each LIT module to override.
   *
   * Only called if the module is reactive, meaning it is visibly on-screen.
   * Clients should override this method as opposed to the base render() method.
   */
  protected renderImpl(): unknown {
    return html``;
  }

  /**
   * A helper method for wrapping async API calls in machinery that a)
   * automatically sets the loading state of the parent widget container and
   * b) ensures that the function only returns the value for the latest async
   * call, and null otherwise;
   */
  async loadLatest<T>(key: string, promise: Promise<T>): Promise<T|null> {
    this.latestLoadPromises.set(key, promise);
    this.setIsLoading(true);

    const result = await promise;

    if (this.latestLoadPromises.get(key) === promise) {
      this.setIsLoading(false);
      this.latestLoadPromises.delete(key);
      return result;
    }

    return null;
  }

  /**
   * Decide if this module should be displayed, based on the current model(s)
   * and dataset.
   */
  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return true;
  }
}

/**
 * A type representing the constructor / class of a LitModule, extended with the
 * static properties that need to be defined on a LitModule.
 */
export type LitModuleType = typeof LitModule&LitModuleClass;
