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

import '../elements/threshold_slider';
// tslint:disable:no-new-decorators
import {customElement, html} from 'lit-element';
import {computed, observable} from 'mobx';
import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {GroupedExamples, ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {doesOutputSpecContain, getMarginFromThreshold, findSpecKeys, isBinaryClassification} from '../lib/utils';
import {ClassificationService, GroupService} from '../services/services';
import {styles as sharedStyles} from './shared_styles.css';


/**
 * A LIT module that renders regression results.
 */
@customElement('thresholder-module')
export class ThresholderModule extends LitModule {
  static title = 'Binary Classifier Thresholds';
  static numCols = 3;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<thresholder-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></thresholder-module>`;
  };

  static get styles() {
    return [sharedStyles];
  }

  // Cost ratio of false positives to false negatives to use in calculating
  // optimal thresholds.
  private costRatio = 1;

  // Selected features to create faceted thresholds from.
  @observable private readonly selectedFacets: string[] = [];

  private readonly classificationService =
    app.getService(ClassificationService);
  private readonly groupService = app.getService(GroupService);

  firstUpdated() {
    const getGroupedExamples = () => this.groupedExamples;
    this.reactImmediately(
        getGroupedExamples, groupedExamples => {
          this.updateMarginCategories(groupedExamples);
        });
  }

  /**
   * Set the facets for which margins can be individually set based on the
   * facet groups selected.
   */
  private updateMarginCategories(groupedExamples: GroupedExamples) {
    for (const predKey of this.binaryClassificationKeys) {
      this.classificationService.setMarginGroups(
          this.model, predKey, groupedExamples);
    }
  }

  @computed
  private get binaryClassificationKeys() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const classificationKeys = findSpecKeys(outputSpec, 'MulticlassPreds');
    return classificationKeys.filter(
        key => isBinaryClassification(outputSpec[key]));
  }

  private async calculateThresholds() {
    const config = {'cost_ratio': this.costRatio};
    const thresholds = await this.apiService.getInterpretations(
        this.appState.currentInputData, this.model,
        this.appState.currentDataset, 'thresholder', config);
    for (const thresholdResults of thresholds) {
      this.classificationService.setMargin(
          this.model, thresholdResults['pred_key'],
          getMarginFromThreshold(thresholdResults['threshold']));
    }
  }

  /** The facet groups created by the feature selector checkboxes. */
  @computed
  private get groupedExamples() {
    // Get the intersectional feature bins.
    const groupedExamples = this.groupService.groupExamplesByFeatures(
        this.appState.currentInputData, this.selectedFacets);
    return groupedExamples;
  }

  renderSliders(predKey: string) {
    const facetKeys = Object.keys(this.groupedExamples);
    const sliders = facetKeys.map(facetKey => {
      const margin = this.classificationService.getMargin(
          this.model, predKey, this.groupedExamples[facetKey]);
      const callback = (e: Event) => {
        this.classificationService.setMargin(
            // tslint:disable-next-line:no-any
            this.model, predKey, (e as any).detail.margin,
            this.groupedExamples[facetKey]);
      };
      return html`<threshold-slider .margin=${margin} label=${facetKey}
                    ?isThreshold=${true} @threshold-changed=${callback}>
                  </threshold-slider>`;
    });
    return html`
        <div class="pred-key-label">${predKey}</div>
        ${sliders}`;
  }

  private renderCheckbox(
      key: string, checked: boolean, onChange: (e: Event, key: string) => void,
      disabled: boolean) {
    // clang-format off
    return html`
        <div class='checkbox-holder'>
          <lit-checkbox
            ?checked=${checked}
            ?disabled=${disabled}
            @change='${(e: Event) => {onChange(e, key);}}'
            label=${key}>
          </lit-checkbox>
        </div>
    `;
    // clang-format on
  }

  renderControls() {
    const handleCostRatioInput = (e: Event) => {
      // tslint:disable-next-line:no-any
      this.costRatio = +((e as any).target.value);
    };
    // Update the selected facets to match the checkboxes.
    const onFeatureCheckboxChange = (e: Event, key: string) => {
      if ((e.target as HTMLInputElement).checked) {
        this.selectedFacets.push(key);
      } else {
        const index = this.selectedFacets.indexOf(key);
        this.selectedFacets.splice(index, 1);
      }
    };

    const costRatioTooltip = "The cost of false positives relative to false " +
        "negatives. Used to find optimal binary classifier thresholds";
    return html`
        <div title=${costRatioTooltip}>Cost ratio (FP/FN):</div>
        <input type=number step="0.01" min=0 max=100 .value=${this.costRatio.toString()}
            @input=${handleCostRatioInput}>
        <label class="cb-label">Facet by</label>
       ${
        this.groupService.denseFeatureNames.map(
            (facetName: string) => this.renderCheckbox(facetName, false,
                (e: Event) => {onFeatureCheckboxChange(e, facetName);}, false))}
        <button class='hairline-button' @click=${this.calculateThresholds}>
          Set optimal threshold
        </button>`;
  }

  render() {
    const sliders =
        this.binaryClassificationKeys.map(key => this.renderSliders(key));
    return html`
        <div class='module-container'>
          <div class='module-toolbar'>
            ${this.renderControls()}
          </div>
          <div class='module-results-area ${SCROLL_SYNC_CSS_CLASS}'>
            ${sliders}
          </div>
        </div>
        `;
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, ['MulticlassPreds'],
                                 isBinaryClassification);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'thresholder-module': ThresholderModule;
  }
}
