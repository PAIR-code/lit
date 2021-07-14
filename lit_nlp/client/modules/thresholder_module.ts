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
import {computed} from 'mobx';
import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {doesOutputSpecContain, getMarginFromThreshold, findSpecKeys, isBinaryClassification} from '../lib/utils';
import {ClassificationService} from '../services/services';
import {styles as sharedStyles} from './shared_styles.css';


/**
 * A LIT module that renders regression results.
 */
@customElement('thresholder-module')
export class ThresholderModule extends LitModule {
  static title = 'Thresholder';
  static numCols = 3;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<thresholder-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></thresholder-module>`;
  };

  static get styles() {
    return [sharedStyles];
  }

  private costRatio = 1;
  private readonly classificationService =
    app.getService(ClassificationService);

  @computed
  private get binaryClassificationKeys() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const classificationKeys = findSpecKeys(outputSpec, 'MulticlassPreds');
    return classificationKeys.filter(
        key => isBinaryClassification(outputSpec[key]));
  }

  private async calculateThresholds() {
    const margins = this.classificationService.marginSettings[this.model] || {};
    const config = {'cost_ratio': this.costRatio};
    const thresholds = await this.apiService.getInterpretations(
        this.appState.currentInputData, this.model,
        this.appState.currentDataset, 'thresholder', config);
    for (const thresholdResults of thresholds) {
      margins[thresholdResults['pred_key']] =
          getMarginFromThreshold(thresholdResults['threshold']);
    }
  }

  renderSlider(key: string) {
    const margins = this.classificationService.marginSettings[this.model] || {};
    const margin = margins[key];
    const callback = (e: Event) => {
      // tslint:disable-next-line:no-any
      margins[(e as any).detail.predKey] = (e as any).detail.margin;
    };
    return html`
        <threshold-slider .margin=${margin} predKey=${key}
                          ?isThreshold=${true} @threshold-changed=${callback}>
        </threshold-slider>`;
  }

  renderControls() {
    const handleCostRatioInput = (e: Event) => {
      // tslint:disable-next-line:no-any
      this.costRatio = +((e as any).target.value);
    };
    const costRatioTooltip = "The cost of false positives relative to false " +
        "negatives. Used to find optimal binary classifier thresholds";
    return html`
        <div title=${costRatioTooltip}>Cost ratio (FP/FN):</div>
        <input type=number step="0.01" min=0 max=100 .value=${this.costRatio.toString()}
            @input=${handleCostRatioInput}>
        <button class='hairline-button' @click=${this.calculateThresholds}>
          Set optimal threshold
        </button>`;
  }

  render() {
    const sliders =
        this.binaryClassificationKeys.map(key => this.renderSlider(key));
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
