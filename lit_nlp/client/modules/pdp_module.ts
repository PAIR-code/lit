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

import '../elements/line_chart';
import '../elements/bar_chart';
// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators';
import {html} from 'lit';
import {until} from 'lit/directives/until';
import {observable} from 'mobx';
import {LitModule} from '../core/lit_module';
import {ExpansionToggle} from '../elements/expansion_panel';
import {CategoryLabel, LitTypeWithNullIdx, LitTypeWithVocab, MulticlassPreds, RegressionScore, Scalar} from '../lib/lit_types';
import {ModelInfoMap, Spec} from '../lib/types';
import {doesInputSpecContain, doesOutputSpecContain, findSpecKeys, setEquals} from '../lib/utils';

import {styles} from './pdp_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

// Dict of possible feature values to model outputs for a given prediction head
// (list for classification, single number for regression).
interface PdpInfo {
  [key: string]: number|number[];
}
// Dict of PdpInfo for all prediction heads of a model.
interface AllPdpInfo {
  [predKey: string]: PdpInfo;
}

// Data for bar or line charts.
type ChartInfo = Map<string|number, number>;

/**
 * A LIT module that renders regression results.
 */
@customElement('pdp-module')
export class PdpModule extends LitModule {
  static override title = 'Partial Dependence Plots';
  static override duplicateForExampleComparison = true;
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <pdp-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </pdp-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  @observable private readonly plotVisibility = new Map<string, boolean>();
  @observable private readonly plotInfo = new Map<string, AllPdpInfo>();

  // Tracks the selected examples for current plots to ensure returned plot info
  // is for the current selection before displaying it.
  private selectionSet = new Set<string>();

  override firstUpdated() {
    const getInputSpec = () => this.appState.getModelSpec(this.model).input;
    this.reactImmediately(getInputSpec, inputSpec => {
      this.resetPlots(inputSpec);
    });

    // When selected data changes, clear the cached plots and set all plots
    // back to hidden.
    this.react(() => this.selectionService.selectedInputData, () => {
      this.selectionSet = new Set(this.selectionService.selectedInputData.map(
          example => example.id));
      this.plotInfo.clear();
      for (const key of this.plotVisibility.keys()) {
        this.plotVisibility.set(key, false);
      }
    });
  }

  // Clear cached plots and find all features for which plots can be generated,
  // setting them to hidden by default.
  private async resetPlots(inputSpec: Spec) {
    this.plotVisibility.clear();
    this.plotInfo.clear();
    const feats = findSpecKeys(inputSpec, [Scalar, CategoryLabel]);
    for (const feat of feats) {
      if (inputSpec[feat].required) {
        this.plotVisibility.set(feat, false);
      }
    }
  }

  // Get plots for a given feature and the selected data from the back-end.
  private async calculatePlotInfo(feat:string) {
    const config = {
      'feature': feat
    };
    const selectedInputs = this.selectionService.selectedInputData;
    const plotInfo = await this.apiService.getInterpretations(
        selectedInputs, this.model, this.appState.currentDataset, 'pdp',
        config);
    return plotInfo;
  }

  renderPlot(feat: string) {
    if (!this.plotVisibility.get(feat)) return html``;

    const spec = this.appState.getModelSpec(this.model);
    const isNumeric = spec.input[feat] instanceof Scalar;

    // Get plot info if already fetched by front-end, or make call to back-end
    // to calcuate it.
    const getPlotInfo = (feat: string): Promise<AllPdpInfo> => {
      const fetchedSelectionSet = new Set(this.selectionSet);

      return new Promise (async (resolved, rejected) => {
        if (this.plotInfo.get(feat) == null) {
          const plotInfo = await this.calculatePlotInfo(feat);
          if (setEquals(fetchedSelectionSet, this.selectionSet)) {
            this.plotInfo.set(feat, plotInfo);
          }
        }
        resolved(this.plotInfo.get(feat)!);
      });
    };

    const renderSpinner = () => {
      return html`
          <div class='pdp-background'>
            <lit-spinner size=${24} color="var(--app-secondary-color)">
            </lit-spinner>
          </div>`;
    };

    const renderPredPlots = (plot: PdpInfo, predKey: string) => {
      const data = new Array<ChartInfo>();
      data.push(new Map<string|number, number>());
      for (const key of Object.keys(plot)) {
        const val = plot[key];
        const xVal = isNumeric ? +key : key;
        if (Array.isArray(val)) {
          for (let i = 0; i < val.length; i++) {
            if (data.length <= i) {
              data.push(new Map<string|number, number>());
            }
            data[i].set(xVal, val[i]);
          }
        } else {
          data[0].set(xVal, val);
        }
      }

      const {vocab} = spec.output[predKey] as LitTypeWithVocab;
      const {null_idx: nullIdx} = spec.output[predKey] as LitTypeWithNullIdx;

      const isClassification = spec.output[predKey] instanceof MulticlassPreds;
      const yRange = isClassification ? [0, 1] : [];
      const renderChart = (chartData: ChartInfo) => {
        if (isNumeric) {
          return html`
              <line-chart height=150 width=300
                  .scores=${chartData} .yScale=${yRange}>
              </line-chart>`;
        } else {
          return html`
              <bar-chart height=150 width=300
                  .scores=${chartData} .yScale=${yRange}>
              </bar-chart>`;
        }

      };
      return html`
          <div class="charts-holder">
            ${data.map((chartData, i) => {
              const label = vocab != null && vocab.length > i ? vocab[i] : '';
              if (nullIdx == null || nullIdx !== i) {
                return html`
                    <div class="chart">
                      <div class="chart-label">${label}</div>
                      ${renderChart(chartData)}
                    </div>`;
              } else {
                return null;
              }
            })}
          </div>`;
    };

    const renderPlotInfo = (plotInfo: AllPdpInfo) => {
      return html`
          <div class='pdp-background'>
            ${Object.keys(plotInfo).map(predKey => {
              return renderPredPlots(plotInfo[predKey], predKey);
            })}
          </div>`;
    };

    // Render spinner until plot info is available, then render the plots.
    return html`${until(getPlotInfo(feat).then(plotInfo => {
      return renderPlotInfo(plotInfo);
    }), renderSpinner())}`;
  }

  renderPlotHolder(feat: string) {
    const expansionHandler = (event: CustomEvent<ExpansionToggle>) => {
      const {isExpanded} = event.detail;
      this.plotVisibility.set(feat, isExpanded);
      this.requestUpdate();
    };

    // clang-format off
    return html`
        <div class='plot-holder'>
          <expansion-panel .label=${feat}
                           @expansion-toggle=${expansionHandler}>
            ${this.renderPlot(feat)}
          </expansion-panel>
        </div>
      `;
    // clang-format on
  }

  override renderImpl() {
    return html`${Array.from(this.plotVisibility.keys()).map(
        feat => this.renderPlotHolder(feat))}`;
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, [RegressionScore, MulticlassPreds])
        && doesInputSpecContain(modelSpecs, [Scalar, CategoryLabel], true);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'pdp-module': PdpModule;
  }
}
