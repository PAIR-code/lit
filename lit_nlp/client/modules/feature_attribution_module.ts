/**
 * @fileoverview Feature attribution info for tabular ML models.
 *
 * @license
 * Copyright 2022 Google LLC
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
import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {FacetsChange} from '../core/faceting_control';
import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {InterpreterClick, InterpreterSettings} from '../elements/interpreter_controls';
import {SortableTemplateResult, TableData} from '../elements/table';
import {FeatureSalience as FeatureSalienceLitType, SingleFieldMatcher} from '../lib/lit_types';
import {IndexedInput, ModelInfoMap} from '../lib/types';
import * as utils from '../lib/utils';
import {findSpecKeys} from '../lib/utils';
import {SignedSalienceCmap} from '../services/color_service';
import {NumericFeatureBins} from '../services/group_service';
import {AppState, GroupService} from '../services/services';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {styles} from './feature_attribution_module.css';

const ALL_DATA = 'Entire Dataset';
const SELECTION = 'Selection';
const LEGEND_INFO_TITLE_SIGNED =
    "Salience is relative to the model's prediction of a class. A positive " +
    "score (more green) for a token means that token influenced the model to " +
    "predict that class, whereas a negaitve score (more pink) means the " +
    "token influenced the model to not predict that class.";

interface AttributionStats {
  min: number;
  median: number;
  max: number;
  mean: number;
}

interface AttributionStatsMap {
  [feature: string]: AttributionStats;
}

interface FeatureSalience {
  salience: {[feature: string]: number};
}

interface FeatureSalienceResult {
  [key: string]: FeatureSalience;
}

interface SummariesMap {
  [facet: string]: AttributionStatsMap;
}

interface VisToggles {
  [name: string]: boolean;
}

/** Aggregate feature attribution for tabular ML models. */
@customElement('feature-attribution-module')
export class FeatureAttributionModule extends LitModule {
  // ---- Static Properties and API ----
  static override title = 'Tabular Feature Attribution';
  static override duplicateForExampleComparison = false;
  static override duplicateAsRow = true;

  static override get styles() {
    return [sharedStyles, styles];
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap) {
    const appState = app.getService(AppState);
    if (appState.metadata == null) return false;

    return Object.values(modelSpecs).some(modelInfo => {
      // The model directly outputs FeatureSalience
      const hasIntrinsicSalience =
          findSpecKeys(modelInfo.spec.output, FeatureSalienceLitType).length >
          0;

      // At least one compatible interpreter outputs FeatureSalience
      const canDeriveSalience = modelInfo.interpreters.some(name => {
        const {metaSpec} = appState.metadata.interpreters[name];
        return findSpecKeys(metaSpec, FeatureSalienceLitType).length > 0;
      });

      return hasIntrinsicSalience || canDeriveSalience;
    });
  }

  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`
      <feature-attribution-module model=${model} .shouldReact=${shouldReact}
        selectionServiceIndex=${selectionServiceIndex}>
      </feature-attribution-module>`;

  // ---- Instance Properties ----

  private readonly groupService = app.getService(GroupService);
  private readonly colorMap = new SignedSalienceCmap();
  private readonly facetingControl = document.createElement('faceting-control');

  @observable private startsOpen?: string;
  @observable private isColored = false;
  @observable private features: string[] = [];
  @observable private bins: NumericFeatureBins = {};
  @observable private readonly settings =
      new Map<string, InterpreterSettings>();
  @observable private summaries: SummariesMap = {};
  @observable private readonly enabled: VisToggles = {
    'model': this.hasIntrinsicSalience,
    [SELECTION]: false
  };

  constructor() {
    super();

    const facetsChange = (event: CustomEvent<FacetsChange>) => {
      this.features = event.detail.features;
      this.bins = event.detail.bins;
    };

    this.facetingControl.contextName = FeatureAttributionModule.title;
    this.facetingControl.addEventListener(
        'facets-change', facetsChange as EventListener);
  }

  @computed
  private get facets() {
    return this.groupService.groupExamplesByFeatures(
        this.bins, this.appState.currentInputData, this.features);
  }

  @computed
  private get hasIntrinsicSalience() {
    if (this.appState.metadata.models[this.model]?.spec?.output) {
      return findSpecKeys(this.appState.metadata.models[this.model].spec.output,
                          FeatureSalienceLitType).length > 0;
    }
    return false;
  }

  @computed
  private get salienceInterpreters() {
    const {interpreters} = this.appState.metadata.models[this.model];
    return Object.entries(app.getService(AppState).metadata.interpreters)
        .filter(
            ([name, i]) => interpreters.includes(name) &&
                findSpecKeys(i.metaSpec, FeatureSalienceLitType).length > 0)
        .map(([name]) => name);
  }

  // ---- Private API ----

  /**
   * Retrieves and summarizes `FeatureSalience` info from the model predictions
   * for the named facet (i.e., subset of data), and adds the summaries to the
   * module's state.
   *
   * Models may provide `FeatureSalience` data in multiple output features.
   * Summmaries are created and stored on state for each feature-facet pair.
   */
  private async predict(facet: string, data: IndexedInput[]) {
    const promise = this.apiService.getPreds(
        data, this.model, this.appState.currentDataset,
        [FeatureSalienceLitType]);
    const results = await this.loadLatest('predictionScores', promise);

    if (results == null) return;

    const outputSpec = this.appState.metadata.models[this.model].spec.output;
    const salienceKeys = findSpecKeys(outputSpec, FeatureSalienceLitType);

    for (const key of salienceKeys) {
      const summaryName = `Feature: ${key} | Facet: ${facet}`;
      const values = results.map(res => res[key]) as FeatureSalience[];
      const summary = this.summarize(values);
      if (summary) {
        this.summaries[summaryName] = summary;

        if (Object.keys(this.summaries).length === 1) {
          this.startsOpen = summaryName;
        }
      }
    }
  }

  /** Updates salience summaries provided by the model. */
  private async updateModelAttributions() {
    if (!this.enabled['model']) return;

    await this.predict(ALL_DATA, this.appState.currentInputData);

    if (this.enabled[SELECTION]) {
      await this.predict(SELECTION, this.selectionService.selectedInputData);
    }

    if (this.features.length) {
      for (const [facet, group] of Object.entries(this.facets)) {
        await this.predict(facet, group.data);
      }
    }
  }

  /**
   * Retrieves and summarizes `FeatureSalience` info from the given interpreter
   * for the named facet (i.e., subset of data), and adds the summaries to the
   * module's state.
   *
   * Interpreters may provide `FeatureSalience` data in multiple output fields.
   * Summmaries are created and stored on state for each field-facet pair.
   */
  private async interpret(name: string, facet: string, data: IndexedInput[]) {
    const runKey = `interpretations-${name}`;
    const {configSpec} = this.appState.metadata.interpreters[name];
    const defaultCallConfig: {[key: string]: unknown} = {};

    for (const [configKey, configInfo] of Object.entries(configSpec)) {
      if (configInfo instanceof SingleFieldMatcher) {
        if (configInfo.default) {
          defaultCallConfig[configKey] = configInfo.default;
        } else if (configInfo.vocab && configInfo.vocab.length) {
          defaultCallConfig[configKey] = configInfo.vocab[0];
        }
      }
    }

    const callConfig = this.settings.get(name) || defaultCallConfig;
    const promise = this.apiService.getInterpretations(
        data, this.model, this.appState.currentDataset, name, callConfig,
        `Running ${name}`);
    const results =
        (await this.loadLatest(runKey, promise)) as FeatureSalienceResult[];

    if (results == null || results.length === 0) return;

    // TODO(b/217724273): figure out a more elegant way to handle
    // variable-named output fields with metaSpec.
    const {metaSpec} = this.appState.metadata.interpreters[name];
    const {output} = this.appState.getModelSpec(this.model);
    const spec = {...metaSpec, ...output};
    const salienceKeys = findSpecKeys(spec, FeatureSalienceLitType);

    for (const key of salienceKeys) {
      if (results[0][key] != null) {
        const salience = results.map((a: FeatureSalienceResult) => a[key]);
        const summaryName =
            `Interpreter: ${name} | Key: ${key} | Facet: ${facet}`;
        this.summaries[summaryName] = this.summarize(salience);
        if (Object.keys(this.summaries).length === 1) {
          this.startsOpen = summaryName;
        }
      }
    }
  }

  /** Updates salience summaries for all enabled interpreters. */
  private async updateInterpreterAttributions() {
    const interpreters = this.salienceInterpreters.filter(i => this.enabled[i]);
    for (const interpreter of interpreters) {
      await this.interpret(interpreter, ALL_DATA,
                           this.appState.currentInputData);

      if (this.enabled[SELECTION]) {
        await this.interpret(interpreter, SELECTION,
                             this.selectionService.selectedInputData);
      }

      if (this.features.length) {
        for (const [facet, group] of Object.entries(this.facets)) {
          await this.interpret(interpreter, facet, group.data);
        }
      }
    }
  }

  private updateSummaries() {
    this.summaries = {};
    this.updateModelAttributions();
    this.updateInterpreterAttributions();
  }

  /**
   * Summarizes the distribution of feature attribution values of a dataset
   */
  private summarize(data: FeatureSalience[]) {
    const statsMap: AttributionStatsMap = {};
    const fields = Object.keys(data[0].salience);

    for (const field of fields) {
      const values = data.map(d => d.salience[field]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const mean = utils.mean(values);
      const median = utils.median(values);
      statsMap[field] = {min, median, max, mean};
    }

    return statsMap;
  }

  private renderSecondaryControls() {
    const change = () => {
      this.isColored = !this.isColored;
    };

    // clang-format off
    return html`
        ${this.facetingControl}
        <span style="felx: 1 1 auto;"></span>
        <lit-checkbox label="Heatmap" ?checked=${this.isColored}
                      @change=${() => {change();}}>
        </lit-checkbox>`;
    // clang-format on
  }

  private renderPrimaryControls() {
    const change = (name: string) => {
      this.enabled[name] = !this.enabled[name];
    };
    const isNoSelectionEmtp = this.selectionService.selectedIds.length === 0;
    const selectionTitle =
        `Attribution is calculated for the entire dataset by default.${
          isNoSelectionEmtp ? ' Make a selection to enable this checkbox.' :
                              ''}`;
    // clang-format off
    return html`
      <lit-checkbox label="Show attributions for selection"
                    title=${selectionTitle} ?disabled=${isNoSelectionEmtp}
                    ?checked=${this.enabled[SELECTION]}
                    @change=${() => {change(SELECTION);}}>
      </lit-checkbox>
      <span style="width: 16px;"></span>
      <span>Show attributions from:</span>
      ${this.salienceInterpreters.map(interp =>
          html`<lit-checkbox label=${interp} ?checked=${this.enabled[interp]}
                             @change=${() => {change(interp);}}>
               </lit-checkbox>`)}`;
    // clang-format on
  }

  private renderInterpreterControls(interpreter: string) {
    const {configSpec, description} =
        this.appState.metadata.interpreters[interpreter];
    const clonedSpec = Object.assign({}, configSpec);
    for (const fieldSpec of Object.values(clonedSpec)) {
      // If the interpreter uses a field matcher, then get the matching field
      // names from the specified spec and use them as the vocab.
      if (fieldSpec instanceof SingleFieldMatcher) {
        fieldSpec.vocab =
            this.appState.getSpecKeysFromFieldMatcher(fieldSpec, this.model);
      }
    }
    const interpreterControlClick = (event: CustomEvent<InterpreterClick>) => {
      this.settings.set(interpreter, event.detail.settings);
      if (!this.enabled[interpreter]) this.enabled[interpreter] = true;
    };
    return html`
      <lit-interpreter-controls @interpreter-click=${interpreterControlClick}
                                .spec=${configSpec} .name=${interpreter}
                                .description=${description || ''}
                                .opened=${this.enabled[interpreter]}>
      </lit-interpreter-controls>`;
  }

  private renderColoredCell(value: number): SortableTemplateResult {
    const txtColor = this.colorMap.textCmap(value);
    const bgColor = this.colorMap.bgCmap(value);
    const styles = styleMap({
      'width': '100%',
      'height': '100%',
      'position': 'relative',
      'text-align': 'right',
      'color': txtColor,
      'background-color': bgColor
    });
    const template = html`<div style=${styles}>${value.toFixed(4)}</div>`;
    return {value, template};
  }

  private renderTable(summary: AttributionStatsMap) {
    const columnNames = [
      {name: 'field', rightAlign: false},
      {name: 'min', rightAlign: true},
      {name: 'median', rightAlign: true},
      {name: 'max', rightAlign: true},
      {name: 'mean', rightAlign: true}
    ];
    const tableData: TableData[] =
        Object.entries(summary).map(([feature, stats]) => {
          const {min, median, max, mean} = stats;
          let fieldsArray: number[] | SortableTemplateResult[] =
              [min, median, max, mean];

          if (this.isColored) {
            fieldsArray = fieldsArray.map(v => this.renderColoredCell(v));
          }

          return [feature, ...fieldsArray];
        });

    // clang-format off
    return html`<lit-data-table .data=${tableData} .columnNames=${columnNames}
                                searchEnabled></lit-data-table>`;
    // clang-format on
  }

  // ---- Public API ----

  override firstUpdated() {
    const dataChange = () => [this.appState.currentInputData, this.features,
                              this.model, Object.values(this.enabled)];
    this.react(dataChange, () => {this.updateSummaries();});

    this.enabled['model'] = this.hasIntrinsicSalience;
    this.updateSummaries();
  }

  override renderImpl() {
    const scale = (val: number) => this.colorMap.bgCmap(val);
    scale.domain = () => this.colorMap.colorScale.domain();

    // clang-format off
    return html`
      <div class='module-container'>
        <div class='module-toolbar'>${this.renderPrimaryControls()}</div>
        <div class='module-results-area'>
          <div class='side-navigation'>
            ${this.salienceInterpreters.map(interpreter =>
                  this.renderInterpreterControls(interpreter))}
          </div>
          <div class='main-content'>
            <div class='module-toolbar'>${this.renderSecondaryControls()}</div>
            <div class='module-results'>
              ${!(Object.keys(this.summaries).length ||
                  this.latestLoadPromises.size)?
                  html`<div style="padding: 8px;">
                          Select a model or interpreter to show attributions.
                          Attributions are calculated from the entire dataset by
                          default, but can also be calculated for the selection
                          or any facets of the entire dataset. Faceting of
                          selections is not supported.
                        </div>`: null}
              ${Object.entries(this.summaries)
                .sort()
                .map(([facet, summary]) => html`
                  <div class="attribution-container">
                    <expansion-panel .label=${facet}
                                     ?expanded=${facet === this.startsOpen}>
                      ${this.renderTable(summary)}
                    </expansion-panel>
                  </div>`)}
              ${this.latestLoadPromises.size ?
                  html`<lit-spinner size=${24} color="var(--lit-cyea-400)">
                       </lit-spinner>`: null}
            </div>
          </div>
        </div>
        <div class="module-footer">
          <div class="color-legend-container">
            <color-legend selectedColorName="Salience" .scale=${scale}
                legendType=${LegendType.SEQUENTIAL} numBlocks=${7}>
            </color-legend>
            <mwc-icon class="icon material-icon-outlined"
                      title=${LEGEND_INFO_TITLE_SIGNED}>
              info_outline
            </mwc-icon>
          </div>
        </div>
      </div>`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'feature-attribution-module': FeatureAttributionModule;
  }
}
