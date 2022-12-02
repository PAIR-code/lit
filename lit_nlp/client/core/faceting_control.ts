/**
 * @fileoverview Element for controlling the faceting behavior of a module.
 *
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

import '../elements/popup_container';
import '../elements/slider';

import {html, TemplateResult} from 'lit';
// tslint:disable:no-new-decorators
import {customElement, property} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {observable} from 'mobx';

import {app} from '../core/app';
import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {getStepSizeGivenRange} from '../lib/utils';
import {FacetingConfig, FacetingMethod, GroupService, NumericFeatureBins} from '../services/group_service';

import {styles} from './faceting_control.css';

const DEFAULT_QUANTILE = 4;
const DEFAULT_BIN_LIMIT = 100;

/** The features and bins that should be used for faceting the dataset. */
export interface FacetsChange {
  features: string[];
  bins: NumericFeatureBins;
}

type FeatureChangeHandler = (event?: Event) => void;

/**
 * Controls for defining faceting behavior.
 *
 * This faceting control should be created using the call
 * `document.createElement('faceting-control')"` or `new FacetingControl()` and
 * stored as a data member of any module that wishes to make use of it, as
 * opposed to being invoked through element templating in a `renderImpl` method
 * call. This is done so that options set in this control aren't reset when a
 * module's render is skipped due to the module being off-screen.
 */
@customElement('faceting-control')
export class FacetingControl extends ReactiveElement {
  private readonly featureConfigs = new Map<string, FacetingConfig>();
  private readonly discreteCount = new Map<string, number>();

  @observable private hasExcessBins = false;
  @observable private features: string[] = [];
  @observable private bins: NumericFeatureBins = {};

  @observable @property({type: Boolean}) disabled = false;
  @observable @property({type: String}) contextName?: string;
  @observable @property({type: Number}) binLimit = DEFAULT_BIN_LIMIT;
  @observable @property({type: Number}) choiceLimit?: number;

  static override get styles() {
    return [sharedStyles, styles];
  }

  constructor(private readonly groupService = app.getService(GroupService)) {
    super();
    this.initFeatureConfigs();
  }

  /**
   * Resets to the default faceting behavior (i.e., do not facet). Typically
   * called if the avaialble features change.
   */
  reset() {
    this.features = [];
    this.updateBins();
    this.initFeatureConfigs();
  }

  /**
   * Resets the faceting configurations to use the Equal Interval binning method
   * and establish the number of bins that would be generated if the user
   * selected the Discrete binning method. Called when the avaiilable numerical
   * features change.
   */
  private initFeatureConfigs() {
    const configs: FacetingConfig[] =
        this.groupService.numericalFeatureNames.map(name => ({
            featureName: name,
            method: FacetingMethod.EQUAL_INTERVAL,
        }));
    const eqaulInterBins = this.groupService.numericalFeatureBins(configs);

    const discreteConfigs: FacetingConfig[] =
        configs.map(c => ({...c, method: FacetingMethod.DISCRETE}));
    const discreteBins =
        this.groupService.numericalFeatureBins(discreteConfigs);

    for (const config of configs) {
      const name = config.featureName;
      config.numBins = Object.keys(eqaulInterBins[name]).length;
      this.featureConfigs.set(name, config);
      this.discreteCount.set(name, Object.keys(discreteBins[name]).length);
    }
  }

  private updateBins() {
    const configs: FacetingConfig[] = this.features
        .filter(f => this.groupService.numericalFeatureNames.includes(f))
        .map(f => this.featureConfigs.get(f) as FacetingConfig);

    this.bins = this.groupService.numericalFeatureBins(configs);

    const count = this.features.reduce((total: number, feature: string) => {
      if (this.groupService.numericalFeatureNames.includes(feature)) {
        return total * Object.keys(this.bins[feature]).length;
      } else if (this.groupService.categoricalFeatureNames.includes(feature)) {
        return total * this.groupService.categoricalFeatures[feature].length;
      } else if (this.groupService.booleanFeatureNames.includes(feature)) {
        return total * 2;
      } else {
        return total;
      }
    }, 1);

    this.hasExcessBins = count > this.binLimit;

    if (!this.hasExcessBins) {
      this.dispatchEvent(new CustomEvent<FacetsChange>('facets-change', {
        detail: {features: this.features, bins: this.bins}
      }));
    }
  }

  override firstUpdated() {
    const numericFeatures = () => this.groupService.denseFeatureNames;
    this.reactImmediately(numericFeatures, () => {this.reset();});
  }

  /**
   * Determines how to render a feature, dispacthing to
   * `renderNumericFeatureOption()` if the feature is numeric, otherwise
   * rendering a simple checkbox and label.
   */
  private renderFeatureOption (feature: string) {
    const checked = this.features.includes(feature);
    const disabled =  this.choiceLimit != null &&
                      this.features.length >= this.choiceLimit && !checked;

    const change: FeatureChangeHandler = () => {
      const activeFacets = [...this.features];

      if (activeFacets.indexOf(feature) !== -1) {
        activeFacets.splice(activeFacets.indexOf(feature), 1);
      } else {
        activeFacets.push(feature);
      }

      this.features = activeFacets;
      this.updateBins();
      this.requestUpdate();
    };

    if (this.groupService.numericalFeatureNames.includes(feature)) {
      return this.renderNumericFeatureOption(feature, checked, disabled,
                                             change);
    }

    const isCategorical =
        this.groupService.categoricalFeatureNames.includes(feature);
    const binCount = isCategorical ?
        this.groupService.categoricalFeatures[feature].length : 2;
    const label = `${feature} (${binCount})`;
    const rowClass = classMap({
      'feature-options-row': true,
      'excess-bins': (this.hasExcessBins && checked)
    });

    // clang-format off
    return html`
        <div class=${rowClass}>
          <lit-checkbox ?checked=${checked} ?disabled=${disabled}
                        @change=${change} label=${label}>
          </lit-checkbox>
        </div>`;
    // clang-format on
  }

  /**
   * Numercial features have multiple configuration options and therefore use
   * their own rendering function to handle updates follwoing user input, i.e.,
   * changing the method, number of bins, or threshold.
   */
  private renderNumericFeatureOption(feature: string, checked: boolean,
                                     disabled: boolean,
                                     change: FeatureChangeHandler) {
    const config = this.featureConfigs.get(feature) as FacetingConfig;

    /** Updates the binning method and bins after user selection. */
    const onMethodChange = (event: Event) => {
      const select = event.target as HTMLSelectElement;
      const method = select.value as FacetingMethod;
      if (config && config.method !== method) {
        config.method = method;

        if (method === FacetingMethod.EQUAL_INTERVAL) {
          config.numBins = 0;   // Reset to inferred bins
          const bins = this.groupService.numericalFeatureBins([config]);
          config.numBins = Object.keys(bins[config.featureName]).length;
        }

        if (method === FacetingMethod.QUANTILE) {
          config.numBins = DEFAULT_QUANTILE;
        }

        if (method === FacetingMethod.THRESHOLD) {
          const [min, max] = this.groupService.numericalFeatureRanges[feature];
          const delta = max - min;
          config.threshold = delta/2 + min;
        }
      }
      this.updateBins();
      this.requestUpdate();
    };

    /** Updates the number of bins for this feature after user interaction. */
    const onNumBinsChange = (event: Event) => {
      const {value} = event.target as HTMLInputElement;
      if (config) {config.numBins = Math.round(Number(value));}
      this.updateBins();
    };

    /** Updates the threshold for this feature after user interaction. */
    const onThresholdChange = (event: Event) => {
      const {value} = event.target as HTMLInputElement;
      if (config) {config.threshold = Number(value);}
      this.updateBins();
    };

    // The user input is determine by the binning method
    let inputField: TemplateResult;

    if (config.method === FacetingMethod.EQUAL_INTERVAL ||
        config.method === FacetingMethod.QUANTILE) {
      /** These methods allow the user to specify the number of bins to use. */
      const value = (config.numBins || DEFAULT_QUANTILE).toString();
      // clang-format off
      inputField = html`
          <input type="number" step="1" min="1" max=${this.binLimit}
                 ?disabled=${disabled} .value=${value}
                 @input=${onNumBinsChange}>`;
      // clang-format on
    } else if (config.method === FacetingMethod.THRESHOLD) {
      /** This method creates two bins given a threshold set by the user. */
      const [min, max] = this.groupService.numericalFeatureRanges[feature];
      const delta = max - min;
      const step = getStepSizeGivenRange(delta);
      const value = (config.threshold || (delta/2 + min)).toString();
      // clang-format off
      inputField = html`
          <input type="number" step=${step} min=${min} max=${max}
                 ?disabled=${disabled} .value=${value}
                 @input=${onThresholdChange}>`;
      // clang-format on
    } else {
      /**
       * This method infers the number of bins to create from the spec, and
       * therefore does not take user input, so we use this div for alignment.
       */
      inputField = html`<div class="no-input"></div>`;
    }

    const rowClass = classMap({
      'feature-options-row': true,
      'excess-bins': (this.hasExcessBins && checked)
    });

    // clang-format off
    return html`
      <div class=${rowClass}>
        <lit-checkbox ?checked=${checked} ?disabled=${disabled}
                      @change=${change} label=${feature}>
        </lit-checkbox>
        <div class="aligner"></div>
        <select class="dropdown" ?disabled=${disabled} @change=${onMethodChange}>
          ${Object.values(FacetingMethod).map(method => {
            const name = (method as string).replace('-', ' ').toUpperCase();
            const selected = method === config.method;
            const discreteCount = this.discreteCount.get(feature) as number;
            const disabled = method === FacetingMethod.DISCRETE &&
                             discreteCount > this.binLimit;
            const label = `${name}${disabled ?
                ` (${discreteCount} bins > ${this.binLimit} max)` : ''}`;
            return html`<option value=${method} ?selected=${selected}
                                ?disabled=${disabled}>${label}</option>`;
          })}
        </select>
        ${inputField}
      </div>`;
    // clang-format on
  }

  private renderFeatureOptions() {
    return this.groupService.denseFeatureNames.map(
        feature => this.renderFeatureOption(feature));
  }

  override render() {
    const facetsList = this.features.length ?
      `${this.features.join(', ')} (${
         this.groupService.numIntersectionsLabel(this.bins, this.features)})` :
      'None';

    const forContext = this.contextName ? ` for ${this.contextName}` : '';
    const title = `Faceting configuration${forContext}`;

    const activeFacetsClass = classMap({
      'active-facets': true,
      'disabled': this.disabled
    });

    // clang-format off
    return html`
      <popup-container>
        <div class="faceting-info" slot='toggle-anchor'>
          <button class="hairline-button" title=${title}
                  ?disabled=${this.disabled}>
            <span class="material-icon">dashboard</span>
            Facets
          </button>
          <div class=${activeFacetsClass}
           @click=${(e: Event) => { e.stopPropagation(); }}>
            : ${facetsList}
          </div>
        </div>
        <div class='config-panel'>
          <div class="panel-header">
            <span class="panel-label">Faceting Config${forContext}</span>
            <span class="choice-limit">(Limit ${this.binLimit} bins ${
              this.choiceLimit != null ? `, ${this.choiceLimit} features` : ''
            })</span>
          </div>
          <div class="panel-options">${this.renderFeatureOptions()}</div>
        </div>
      </popup-container>`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'faceting-control': FacetingControl;
  }
}
