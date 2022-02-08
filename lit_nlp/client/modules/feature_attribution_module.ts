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
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {FacetsChange} from '../core/faceting_control';
import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {IndexedInput, ModelInfoMap} from '../lib/types';
import * as utils from '../lib/utils';
import {findSpecKeys} from '../lib/utils';
import {AppState, GroupService} from '../services/services';
import {NumericFeatureBins} from '../services/group_service';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {styles} from './feature_attribution_module.css';

const ALL_DATA = 'Entire Dataset';

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
           findSpecKeys(modelInfo.spec.output, 'FeatureSalience').length > 0;

      // At least one compatible interpreter outputs FeatureSalience
      const canDeriveSalience = modelInfo.interpreters.some(name => {
        const {metaSpec} = appState.metadata.interpreters[name];
        return findSpecKeys(metaSpec, 'FeatureSalience').length > 0;
      });

      return hasIntrinsicSalience || canDeriveSalience;
    });
  }

  static override template(model = '') {
    // clang-format off
    return html`<feature-attribution-module model=${model}>
                </feature-attribution-module>`;
    // clang format on
  }

  // ---- Instance Properties ----

  private readonly groupService = app.getService(GroupService);

  @observable private startsOpen?: string;
  @observable private features: string[] = [];
  @observable private bins: NumericFeatureBins = {};
  @observable private summaries: SummariesMap = {};
  @observable private readonly enabled: VisToggles = {
    'model': this.hasIntrinsicSalience
  };

  @computed
  private get facets() {
    return this.groupService.groupExamplesByFeatures(
        this.bins, this.appState.currentInputData, this.features);
  }

  @computed
  private get hasIntrinsicSalience() {
    if (this.appState.metadata.models[this.model]?.spec?.output) {
      return findSpecKeys(this.appState.metadata.models[this.model].spec.output,
                          'FeatureSalience').length > 0;
    }
    return false;
  }

  @computed
  private get salienceInterpreters() {
    const {interpreters} = this.appState.metadata.models[this.model];
    return Object.entries(app.getService(AppState).metadata.interpreters)
                 .filter(([name, i]) =>
                    interpreters.includes(name) &&
                    findSpecKeys(i.metaSpec,'FeatureSalience').length > 0)
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
        data, this.model, this.appState.currentDataset, ['FeatureSalience']);
    const results = await this.loadLatest('predictionScores', promise);

    if (results == null) return;

    const outputSpec = this.appState.metadata.models[this.model].spec.output;
    const salienceKeys = findSpecKeys(outputSpec, 'FeatureSalience');

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
    const promise = this.apiService.getInterpretations(
        data, this.model, this.appState.currentDataset, name, {},
        `Running ${name}`);
    const results =
        (await this.loadLatest(runKey, promise)) as FeatureSalienceResult[];

    if (results == null || results.length === 0) return;

    // TODO(b/217724273): figure out a more elegant way to handle
    // variable-named output fields with metaSpec.
    const {metaSpec} = this.appState.metadata.interpreters[name];
    const {output} = this.appState.getModelSpec(this.model);
    const spec = {...metaSpec, ...output};
    const salienceKeys = findSpecKeys(spec, 'FeatureSalience');

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

  private renderFacetControls() {
    const updateFacets = (event: CustomEvent<FacetsChange>) => {
      this.features = event.detail.features;
      this.bins = event.detail.bins;
    };

    // clang-format off
    return html`<faceting-control @facets-change=${updateFacets}
                                  contextName=${FeatureAttributionModule.title}>
                </faceting-control>`;
    // clang-format on
  }

  private renderSalienceControls() {
    const change = (name: string) => {
      this.enabled[name] = !this.enabled[name];
    };
    // clang-format off
    return html`
      <span>Show attributions from:</span>
      ${this.hasIntrinsicSalience ?
          html` <lit-checkbox label=${this.model}
                              ?checked=${this.enabled['model']}
                              @change=${() => {change('model');}}>
                </lit-checkbox>` : null}
      ${this.salienceInterpreters.map(interp =>
          html`<lit-checkbox label=${interp} ?checked=${this.enabled[interp]}
                             @change=${() => {change(interp);}}>
               </lit-checkbox>`)}`;
    // clang-format on
  }

  private renderTable(summary: AttributionStatsMap) {
    const columnNames = ['field', 'min', 'median', 'max', 'mean'];
    const tableData: TableData[] =
        Object.entries(summary).map(([feature, stats]) => {
          const {min, median, max, mean} = stats;
          return [feature, min, median, max, mean];
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

  override render() {
    // clang-format off
    return html`
      <div class='module-container'>
        <div class='module-toolbar'>${this.renderSalienceControls()}</div>
        <div class='module-toolbar'>${this.renderFacetControls()}</div>
        <div class='module-results-area'>
          ${Object.entries(this.summaries)
            .sort()
            .map(([facet, summary]) => html`
              <div class="attribution-container">
                <expansion-panel .label=${facet}
                                 ?expanded=${facet === this.startsOpen}>
                  ${this.renderTable(summary)}
                </expansion-panel>
              </div>`)}
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
