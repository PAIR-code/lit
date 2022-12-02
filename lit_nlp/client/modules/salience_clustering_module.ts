/**
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

import '../elements/expansion_panel';

import {html} from 'lit';
// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators';
import {styleMap} from 'lit/directives/style-map';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {InterpreterClick} from '../elements/interpreter_controls';
import {SortableTemplateResult, TableData} from '../elements/table';
import {CategoryLabel, FieldMatcher, TokenSalience} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, IndexedInput, Input, ModelInfoMap} from '../lib/types';
import {cloneSpec, createLitType, findSpecKeys} from '../lib/utils';
import {ColumnData} from '../services/data_service';
import {DataService} from '../services/services';

import {styles} from './salience_clustering_module.css';

const SALIENCE_MAPPER_KEY = 'Salience Mapper';
const N_CLUSTERS_KEY = 'Number of Clusters';
const SALIENCE_CLUSTERING_INTERPRETER_NAME = 'Salience Clustering';
const RESULT_TOP_TOKENS = 'Top Tokens';
const REUSE_CLUSTERING = 'reuse_clustering';

interface ModuleState {
  dataColumns: string[];
  clusteringConfig: CallConfig;
  salienceConfigs: {[salienceMapper: string]: CallConfig};
  clusteringState: ClusteringState;
}

// Aggregated result of the clustering interpreter.
interface ClusteringState {
  clusterInfos: {[gradKey: string]: ClusterInfo[]};
  topTokenInfosByClusters: {[gradKey: string]: TopTokenInfosByCluster[]};
  isLoading: boolean;
}

// Clustering result for a single piece of text.
interface ClusterInfo {
  example: Input;
  clusterId: number;
}

// Top token information per cluster, sorted by the ascending by cluster IDs.
interface TopTokenInfosByCluster {
  topTokenInfos: TopTokenInfo[];
}

// Data for 1 top token instance.
interface TopTokenInfo {
  token: string;
  weight: number;
}

// Indicators which expandable areas are open or closed.
interface VisToggles {
  [name: string]: boolean;
}

/**
 * A LIT module that renders salience clustering results.
 */
@customElement('salience-clustering-module')
export class SalienceClusteringModule extends LitModule {
  static override title = 'Salience Clustering Results';
  static override numCols = 1;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <salience-clustering-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </salience-clustering-module>`;

  private readonly dataService = app.getService(DataService);
  private runCount: number = 0;
  private statusMessage: string = '';

  static override get styles() {
    return [sharedStyles, styles];
  }

  // Mapping from salience mapper to clustering results.
  @observable private state: ModuleState = {
      dataColumns: [],
      clusteringConfig: {},
      salienceConfigs: {},
      clusteringState: {
        clusterInfos: {},
        topTokenInfosByClusters: {},
        isLoading: false,
      },
    };

  @observable private readonly expanded: VisToggles = {
    [RESULT_TOP_TOKENS]: true,
  };

  override firstUpdated() {
    const state: ModuleState = {
      dataColumns: [],
      clusteringConfig: {},
      salienceConfigs: {},
      clusteringState: {
        clusterInfos: {},
        topTokenInfosByClusters: {},
        isLoading: false,
      },
    };

    this.state = state;
  }

  private async runInterpreter() {
    const salienceMapper = this.state.clusteringConfig[SALIENCE_MAPPER_KEY];
    this.state.clusteringState.clusterInfos = {};
    const inputs = this.selectionService.selectedOrAllInputData;
    if (!this.canRunClustering) {
      return;
    }
    this.state.clusteringState.isLoading = true;
    const clusteringResult = await this.apiService.getInterpretations(
        inputs,
        this.model,
        this.appState.currentDataset,
        SALIENCE_CLUSTERING_INTERPRETER_NAME,
        {...this.state.clusteringConfig,
            ...this.state.salienceConfigs[salienceMapper]},
        `Running ${salienceMapper}`);
    this.state.clusteringState.isLoading = false;
    this.state.dataColumns = Object.keys(inputs[0].data);
    const dataType = createLitType(CategoryLabel);
    dataType.vocab = Array.from(
        {length: this.state.clusteringConfig[N_CLUSTERS_KEY]},
        (value, key) => key.toString()
    );

    // Function to get value for this new data column when new datapoints are
    // added.
    const getValueFn = async (gradKey: string, input: IndexedInput) => {
      const config = {
          ...this.state.clusteringConfig,
          ...this.state.salienceConfigs[salienceMapper],
          [REUSE_CLUSTERING]: true,
      };
      const clusteringResult = await this.apiService.getInterpretations(
          [input],
          this.model,
          this.appState.currentDataset,
          SALIENCE_CLUSTERING_INTERPRETER_NAME,
          config,
          `Running ${salienceMapper}`);
      return clusteringResult['cluster_ids'][gradKey][0].toString();
    };

    for (const gradKey of Object.keys(clusteringResult['cluster_ids'])) {
      // Store per-example cluster IDs.
      const clusterInfos: ClusterInfo[] = [];
      const clusterIds = clusteringResult['cluster_ids'][gradKey];
      const featName = 'Cluster IDs: ' +
          `${this.state.clusteringConfig[SALIENCE_MAPPER_KEY]}, ` +
          `${this.state.clusteringConfig[N_CLUSTERS_KEY]}, ${gradKey}, ` +
          `${this.runCount}`;
      const dataMap: ColumnData = new Map();

      for (let i = 0; i < clusterIds.length; i++) {
        const clusterInfo: ClusterInfo = {
            example: inputs[i].data, clusterId: clusterIds[i]};
        clusterInfos.push(clusterInfo);
        dataMap.set(inputs[i].id, clusterInfo.clusterId.toString());
      }
      this.state.clusteringState.clusterInfos[gradKey] = clusterInfos;
      const localGetValueFn = (input: IndexedInput) =>
          getValueFn(gradKey, input);
      this.dataService.addColumn(
          dataMap, SALIENCE_CLUSTERING_INTERPRETER_NAME, featName, dataType,
          'Interpreter', localGetValueFn);
      this.statusMessage = `New column added: ${featName}.`;

      // Store top tokens.
      this.state.clusteringState.topTokenInfosByClusters[gradKey] = [];
      const topTokenInfosByClusters = clusteringResult['top_tokens'][gradKey];
      const clusterCount = topTokenInfosByClusters.length;

      for (let clusterId = 0; clusterId < clusterCount; clusterId++) {
        const topTokenInfosByCluster: TopTokenInfosByCluster = {
            topTokenInfos: []};

        for (const topTokenTuple of topTokenInfosByClusters[clusterId]) {
          topTokenInfosByCluster.topTokenInfos.push(
              {token: topTokenTuple[0], weight: topTokenTuple[1]});
        }
        this.state.clusteringState.topTokenInfosByClusters[gradKey].push(
            topTokenInfosByCluster);
      }
    }
    this.runCount++;
  }

  renderSpinner() {
    // clang-format off
    return html`
      <div class="spinner-container">
        <lit-spinner size=${24} color="var(--app-secondary-color)">
        </lit-spinner>
      </div>`;
    // clang format on
  }

  renderDataTable() {
    const renderSingleGradKeyClusterInfos =
        (gradKey: string, clusterInfos: ClusterInfo[]) => {
          const rows: TableData[] = clusterInfos.map((clusterInfo) => {
            const row: string[] = [];
            for (const dataColumn of this.state.dataColumns) {
              row.push(clusterInfo.example[dataColumn]);
            }
            row.push(clusterInfo.clusterId.toString());
            return row;
          });

          // clang-format off
          return html`
            <div class="grad-key-row">
              <div class="grad-key-label">${gradKey}</div>
              <lit-data-table
                .columnNames=${
                    [...this.state.dataColumns, 'cluster ID']}
                .data=${rows}
                searchEnabled
                selectionEnabled
                paginationEnabled
              ></lit-data-table>
            </div>`;
          // clang format on
        };

    const renderClusterInfos =
        (gradKeyClusterInfos: {[name: string]: ClusterInfo[]}) => {
          const gradKeys = Object.keys(gradKeyClusterInfos);
          // clang-format off
          return html`
              ${
              gradKeys.map(
                  gradKey => renderSingleGradKeyClusterInfos(
                      gradKey, gradKeyClusterInfos[gradKey]))}
              ${
              this.state.clusteringState.isLoading ? this.renderSpinner() :
                                                     null}`;
          // clang format on
        };

    return renderClusterInfos(this.state.clusteringState.clusterInfos);
  }

  // Render a table that contains all top tokens and their weights per cluster.
  private renderSingleGradKeyTopTokenInfos(gradKey: string,
      topTokenInfosByClusters: TopTokenInfosByCluster[]) {
    const clusterCount = topTokenInfosByClusters.length;
    const maxTopTokenCount = Math.max(...topTokenInfosByClusters.map(
        topTokenInfosByCluster => topTokenInfosByCluster.topTokenInfos.length
    ));

    const columnNames: string[] = [];

    for (let clusterId = 0; clusterId < clusterCount; clusterId++) {
      columnNames.push(`Cluster ${clusterId}`);
    }

    const rows: TableData[] = [];
    const tokenWeightStyle = styleMap({
      'display': 'flex',
      'flex-direction': 'row',
      'justify-content': 'space-between',
      'width': '100%',
    });

    for (let exampleIdx = 0; exampleIdx < maxTopTokenCount;
          exampleIdx++) {
      const row: SortableTemplateResult[] = [];

      for (let clusterId = 0; clusterId < clusterCount; clusterId++) {
        const topTokenInfos = topTokenInfosByClusters[clusterId];
        if (topTokenInfos.topTokenInfos.length < maxTopTokenCount) {
          row.push({template: html``, value: 0});
        } else {
          const {token, weight} = topTokenInfos.topTokenInfos[exampleIdx];
          row.push({
            template: html`
              <div style=${tokenWeightStyle}>
                <span>${token}</span>
                <span>(${weight.toFixed(2)})</span>
              </div>
            `,
            value: weight
          });
        }
      }
      rows.push(row);
    }

    // clang-format off
    return html`
      <div class="grad-key-row">
        <div class="grad-key-label">${gradKey}</div>
        <lit-data-table
          .columnNames=${columnNames}
          .data=${rows}
          searchEnabled
          selectionEnabled
          paginationEnabled
        ></lit-data-table>
      </div>`;
    // clang format on
  }

  renderTopTokens() {
    const gradKeyTopTokenInfos =
        this.state.clusteringState.topTokenInfosByClusters;
    const gradKeys = Object.keys(gradKeyTopTokenInfos);
    // clang-format off
    return html`
      ${gradKeys.map(
          gradKey => this.renderSingleGradKeyTopTokenInfos(
              gradKey, gradKeyTopTokenInfos[gradKey]))}
      ${this.state.clusteringState.isLoading ? this.renderSpinner() : null}`;
    // clang format on
  }

  private getSalienceInterpreterNames() {
    const names: string[] = [];
    const {interpreters} = this.appState.metadata;
    const validInterpreters =
        this.appState.metadata.models[this.model].interpreters;
    for (const key of validInterpreters) {
      const salienceKeys = findSpecKeys(
          interpreters[key].metaSpec, TokenSalience);
      if (salienceKeys.length !== 0) {
        names.push(key);
      }
    }
    return names;
  }

  renderControlsAndResults() {
    if (this.state.clusteringState == null ||
        this.state.clusteringState.clusterInfos == null) {
      return html`<div>Nothing to show.</div>`;
    }

    const moduleControlsApplyCallback =
        (event: CustomEvent<InterpreterClick>) => {
          this.state.clusteringConfig = event.detail.settings;
          this.runInterpreter();
        };

    const methodControlsApplyCallback =
        (event: CustomEvent<InterpreterClick>) => {
          const {name, settings} =  event.detail;
          this.state.salienceConfigs[name] = settings;
        };

    // clang-format off
    const renderInterpreterControls = (name: string) => {
      const spec = this.appState.metadata.interpreters[name].configSpec;
      const clonedSpec = cloneSpec(spec);
      for (const fieldSpec of Object.values(clonedSpec)) {
        // If the generator uses a field matcher, then get the matching
        // field names from the specified spec and use them as the vocab.
        if (fieldSpec instanceof FieldMatcher) {
          fieldSpec.vocab = this.appState.getSpecKeysFromFieldMatcher(
              fieldSpec, this.model);
        }
      }
      if (Object.keys(clonedSpec).length === 0) {
        return;
      }
      return html`
        <lit-interpreter-controls
          .spec=${clonedSpec}
          .name=${name}
          .opened=${name === SALIENCE_CLUSTERING_INTERPRETER_NAME}
          @interpreter-click=${
              name === SALIENCE_CLUSTERING_INTERPRETER_NAME ?
                  moduleControlsApplyCallback :
                  methodControlsApplyCallback}>
        </lit-interpreter-controls>`;
    };
    // Always show the clustering config first.
    const interpreters: string[] = [SALIENCE_CLUSTERING_INTERPRETER_NAME,
                                    ...this.getSalienceInterpreterNames()];
    const expansionArea = (resultName: string) => {
      let content = html``;

      if (resultName === RESULT_TOP_TOKENS) {
        content = this.renderTopTokens();
      }
      return html`
          <expansion-panel
              .label=${resultName}
              ?expanded=${this.expanded[resultName]}>
            ${content}
          </expansion-panel>`;
    };
    return html`
      <div class="controls-and-results-container">
        <div class="clustering-controls-container">
          ${interpreters.map(renderInterpreterControls)}
        </div>
        <div class="clustering-results-container">
          ${expansionArea(RESULT_TOP_TOKENS)}
        </div>
      </div>`;
    // clang-format on
  }

  private get canRunClustering() {
    const inputs = this.selectionService.selectedOrAllInputData;
    return (inputs != null && inputs.length >= 2);
  }

  private renderSelectionWarning() {
    // clang-format off
    return html`
      <div class="selection-warning">
        Please select no datapoint (for entire dataset) or >= 2 datapoints to
        compute clusters.
      </div>`;
    // clang format on
  }

  override renderImpl() {
    // clang-format off
    return html`
      <div class='module-container'>
        <div class='module-results-area'>
          ${this.renderControlsAndResults()}
        </div>
        <div class="module-footer">
          <p class="module-status">
            ${this.canRunClustering ? null : this.renderSelectionWarning()}
          </p>
          <p class="module-status">${this.statusMessage}</p>
        </div>
      </div>`;
    // clang format on

  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap) {
    return Object.values(modelSpecs).some(modelInfo => modelInfo.interpreters
        .indexOf(SALIENCE_CLUSTERING_INTERPRETER_NAME) !== -1);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'salience-clustering-module': SalienceClusteringModule;
  }
}
