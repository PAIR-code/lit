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
import '../elements/expansion_panel';
import '../elements/line_chart';
import {customElement} from 'lit/decorators';
import {html, TemplateResult} from 'lit';
import {action, computed, observable} from 'mobx';
import {FacetsChange} from '../core/faceting_control';
import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {MulticlassPreds} from '../lib/lit_types';
import {GroupedExamples, IndexedInput, ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, hasParent} from '../lib/utils';
import {GroupService} from '../services/services';
import {NumericFeatureBins} from '../services/group_service';
import {styles} from './curves_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

// Response from backend curves interpreter.
interface CurvesResponse {
  // Using case to achieve parity with the property names in Python code
  // tslint:disable-next-line:enforce-name-casing
  pr_data: number[][];
  // tslint:disable-next-line:enforce-name-casing
  roc_data: number[][];
}

// Data to send to line-chart element.
interface CurvesData {
  prCurve: Map<number, number>;
  rocCurve: Map<number, number>;
}

/**
 * A LIT module that renders PR/ROC curves.
 */
@customElement('curves-module')
export class CurvesModule extends LitModule {
  static override title = 'PR/ROC Curves';
  static override numCols = 3;
  static override duplicateForModelComparison = false;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => {
        return html`
  <curves-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </curves-module>`;
      };

  static override get styles() {
    return [
      sharedStyles, styles
    ];
  }

  @observable private readonly isPanelCollapsed = new Map();
  @observable private datasetCurves?: CurvesData[];
  @observable private groupedCurves: {[group: string]: CurvesData[]} = {};
  @observable private selectedPredKeyIndex = 0;
  @observable private selectedPositiveLabelIndex = 0;

  private selectedFacetBins: NumericFeatureBins = {};

  // Selected features to create faceted thresholds from.
  @observable private readonly selectedFacets: string[] = [];

  private readonly groupService = app.getService(GroupService);
  private readonly facetingControl = document.createElement('faceting-control');

  constructor() {
    super();

    const facetsChange = (event: CustomEvent<FacetsChange>) => {
      this.setFacetInfo(event);
    };
    this.facetingControl.contextName = CurvesModule.title;
    this.facetingControl.addEventListener(
        'facets-change', facetsChange as EventListener);
  }

  override firstUpdated() {

    this.reactImmediately(
        () => [this.appState.currentInputData, this.predKey,
               this.positiveLabel], async () => {
      this.datasetCurves = await this.getCurveData(
          this.appState.currentInputData, this.predKey,
          this.positiveLabel);
    });
    this.reactImmediately(
        () => [this.groupedExamples, this.predKey,
               this.positiveLabel], async () => {
          await this.getGroupedCurveData(this.groupedExamples, this.predKey,
                                         this.positiveLabel);
        });
  }

  @action
  private async getGroupedCurveData(groupedExamples: GroupedExamples,
                                    predKey: string, positiveLabel: string) {
    this.groupedCurves = {};
    for (const facets of Object.keys(groupedExamples)) {
      if (facets === '') {
        continue;
      }
      this.groupedCurves[facets] = await this.getCurveData(
          groupedExamples[facets].data, predKey, positiveLabel);
    }
  }

  private async getCurveData(
      data: IndexedInput[], predKey: string, positiveLabel: string) {
    const models = this.appState.currentModels;
    return Promise.all(models.map(
        async (model: string) => this.makeBackendCall(
            data, model, predKey, positiveLabel)));
  }

  private async makeBackendCall(
      data: IndexedInput[], model: string, predKey: string,
      positiveLabel: string) {

    if (this.appState.getModelSpec(model).output[predKey] == null) {
      return {
        prCurve: new Map<number, number>(),
        rocCurve: new Map<number, number>()
      };
    }
    const config = {
      'Label': positiveLabel,
      'Prediction field': predKey
    };
    const curveResponse = await this.apiService.getInterpretations(
        data, model,
        this.appState.currentDataset, 'curves', config);
    return this.convertCurveResponse(curveResponse);
  }

  private convertCurveResponse(curves: CurvesResponse) {
    const data: CurvesData = {
      prCurve: new Map<number, number>(),
      rocCurve: new Map<number, number>()
    };

    // If first entry in either plot isn't at x=0, then add the appropriate
    // first point at x=0 for the plot type.
    for (const entry of curves.pr_data) {
      if (data.prCurve.size === 0 && entry[0] !== 0) {
        data.prCurve.set(0, 1);
      }
      data.prCurve.set(entry[0], entry[1]);
    }
    for (const entry of curves.roc_data) {
      if (data.prCurve.size === 0 && entry[0] !== 0) {
        data.prCurve.set(0, 0);
      }
      data.rocCurve.set(entry[0], entry[1]);
    }
    return data;
  }

  @computed
  get predKeyOptions() {
    return this.appState.currentModels.flatMap((modelName: string) => {
      const modelSpec = this.appState.metadata.models[modelName].spec;
      return findSpecKeys(modelSpec.output, MulticlassPreds);
    });
  }

  @computed
  get predKey() {
    return this.predKeyOptions[this.selectedPredKeyIndex];
  }

  @computed
  get positiveLabelOptions(): string[] {
    for (const modelName of this.appState.currentModels) {
      const modelSpec = this.appState.metadata.models[modelName].spec;
      if (this.predKey in modelSpec.output) {
        const fieldSpec = modelSpec.output[this.predKey];
        if (fieldSpec instanceof MulticlassPreds) {
          // Find default positive label by finding first index that isn't
          // the null index for a predicted class vocab.
          this.selectedPositiveLabelIndex =
              fieldSpec.vocab.findIndex((elem, i) => i !== fieldSpec.null_idx);

          return fieldSpec.vocab;
        }
      }
    }
    return [];
  }

  @computed
  get positiveLabel() {
    return this.positiveLabelOptions[this.selectedPositiveLabelIndex];
  }

  /** The facet groups created by the faceting controls. */
  @computed
  private get groupedExamples() {
    // Get the intersectional feature bins.
    const groupedExamples = this.groupService.groupExamplesByFeatures(
        this.selectedFacetBins,
        this.appState.currentInputData,
        this.selectedFacets);
    return groupedExamples;
  }

  renderPredKeySelect() {
    const options = this.predKeyOptions;
    const htmlOptions = options.map(predKey => {
      return html`
        <option value=${predKey}>${predKey}</option>
      `;
    });

    const handleChange = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      this.selectedPredKeyIndex = select?.selectedIndex || 0;
    };

    return this.renderSelect(
        'Prediction key', htmlOptions, handleChange, options[0]);
  }

  renderPositiveLabelSelect() {
    const options = this.positiveLabelOptions;
    const renderOption = (label: string, selected: boolean) =>
      html`<option value=${label} ?selected=${selected}>${label}</option>`;
    const htmlOptions = options.map((label, i) =>
        renderOption(label, i === this.selectedPositiveLabelIndex));

    const handleChange = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      this.selectedPositiveLabelIndex = select?.selectedIndex || 0;
    };

    const defaultValue = this.positiveLabelOptions[
        this.selectedPositiveLabelIndex];
    return this.renderSelect(
        'Positive label', htmlOptions, handleChange, defaultValue);
  }

  renderSelect(
      label: string, options: TemplateResult[], onChange: (e: Event) => void,
      defaultValue: string) {
    return html`
      <div class="dropdown-container">
        <label class="dropdown-label">${label}:</label>
        <select class="dropdown" @change=${onChange} .value=${defaultValue}>
          ${options}
        </select>
      </div>
    `;
  }

  private renderChart(data: Map<number, number>, title: string,
                      xRange: number[], yRange: number[]) {
    return html`
        <div class="chart-holder">
          <div class="chart-title">${title}</div>
          <line-chart height=150 width=250
                      .scores=${data} .xScale=${xRange} .yScale=${yRange}>
          </line-chart>
        </div>`;
  }

  private renderCharts(
      curves: CurvesData[], title: string, isFirstPanel: boolean) {
    const toggleCollapse = () => {
      const isHidden = (this.isPanelCollapsed.get(title) == null) ?
          !isFirstPanel : this.isPanelCollapsed.get(title);
      this.isPanelCollapsed.set(title, !isHidden);
    };
    // This plot's value in isPanelCollapsed gets set in toggleCollapse and is
    // null before the user opens/closes it for the first time. This uses the
    // collapseByDefault setting if isPanelCollapsed hasn't been set yet.
    const isCollapsed = (this.isPanelCollapsed.get(title) == null) ?
        !isFirstPanel : this.isPanelCollapsed.get(title);

    const xRange = [0, 1];
    const yRange = [0, 1];
    // TODO(b/228963945) - Combine model-specific charts into one chart with all
    // models as separate lines.

    // clang-format off
    return html`
      <expansion-panel .label=${title} padLeft padRight
          @expansion-toggle=${toggleCollapse} ?expanded=${!isCollapsed}>
        <div class="charts-holder">
          ${curves.map((curvesData, i) => {
            let fullTitle = `Precision / Recall`;
            if (curves.length > 1) {
              fullTitle += ` - ${this.appState.currentModels[i]}`;
            }
            return this.renderChart(
                curvesData.prCurve, fullTitle, xRange, yRange);
          })}
          ${curves.map((curvesData, i) => {
            let fullTitle = `ROC (TPR / FPR)`;
            if (curves.length > 1) {
              fullTitle += ` - ${this.appState.currentModels[i]}`;
            }
            return this.renderChart(
                curvesData.rocCurve, fullTitle, xRange, yRange);
          })}
        </div>
      </expansion-panel>`;
    // clang-format on
  }

  @action
  private setFacetInfo(event: CustomEvent<FacetsChange>) {
    this.selectedFacets.length = 0;
    this.selectedFacets.push(...event.detail.features);
    this.selectedFacetBins = event.detail.bins;
  }

  private renderHeader() {
    // Render facet control, predKey dropdown, and positive label dropdown.
    // clang-format off
    return html`
        ${this.renderPredKeySelect()}
        ${this.renderPositiveLabelSelect()}
        ${this.facetingControl}
    `;
    // clang-format on
  }

  override renderImpl() {
    const groups = Object.keys(this.groupedCurves);

    // clang-format off
    return html`
        <div class='module-container'>
          <div class='module-toolbar'>
            ${this.renderHeader()}
          </div>
          <div class='module-results-area ${SCROLL_SYNC_CSS_CLASS}'>
            ${this.datasetCurves ?
                this.renderCharts(this.datasetCurves, 'Dataset', true) : null}
            ${groups.map(group =>
                this.renderCharts(this.groupedCurves[group], group, false)
            )}
          </div>
        </div>
        `;
    // clang-format on
  }

  static override shouldDisplayModule(
      modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, MulticlassPreds, hasParent);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'curves-module': CurvesModule;
  }
}
