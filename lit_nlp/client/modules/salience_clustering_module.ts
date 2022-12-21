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
import {LegendType} from '../elements/color_legend';
import {InterpreterClick} from '../elements/interpreter_controls';
import {TableData} from '../elements/table';
import {CategoryLabel, FieldMatcher, TokenSalience} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, IndexedInput, ModelInfoMap} from '../lib/types';
import {cloneSpec, createLitType, findSpecKeys} from '../lib/utils';
import {SalienceCmap, UnsignedSalienceCmap} from '../services/color_service';
import {ColumnData} from '../services/data_service';
import {DataService} from '../services/services';

import {styles} from './salience_clustering_module.css';

const SALIENCE_MAPPER_KEY = 'Salience Mapper';
const N_CLUSTERS_KEY = 'Number of Clusters';
const SALIENCE_CLUSTERING_INTERPRETER_NAME = 'Salience Clustering';
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

// Clustering assignment result for each data point.
interface ClusterInfo {
  exampleId: string;
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
        const clusterInfo:
            ClusterInfo = {exampleId: inputs[i].id, clusterId: clusterIds[i]};
        clusterInfos.push(clusterInfo);
        dataMap.set(inputs[i].id, clusterInfo.clusterId.toString());
      }
      this.state.clusteringState.clusterInfos[gradKey] = clusterInfos;
      const localGetValueFn = (input: IndexedInput) =>
          getValueFn(gradKey, input);
      this.dataService.addColumn(
          dataMap, SALIENCE_CLUSTERING_INTERPRETER_NAME, featName, dataType,
          'Interpreter', localGetValueFn);
      const numDataPoints = clusterInfos.length;
      this.statusMessage = `Clustered ${numDataPoints} datapoints ` +
          `(Generated column name: ${featName}).`;

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
    // clang-format on
  }

  // Render token chip.
  // TODO(b/204887716): This needs to be replaced with a custom element.
  private renderToken(token: string, weight: number, cmap: SalienceCmap,
    gradKey: string) {
    const tokenStyle = styleMap({
      'background-color': cmap.bgCmap(weight),
      'border-radius': '2px',
      'color': cmap.textCmap(weight),
      'margin': '2px 5px',
      'padding': '1px 3px'
    });

    return html`
      <div class="salient-token-for-cluster" style=${tokenStyle}
        title=${weight.toPrecision(3)} data-gradkey=${gradKey}>
        ${token}
      </div>`;
  }

  // Render a table that lists clusters with their top tokens.
  private renderSingleGradKeyTopTokenInfos(
      gradKey: string, topTokenInfosByClusters: TopTokenInfosByCluster[],
      clusterInfos: ClusterInfo[]) {
    const unsignedCmap = new UnsignedSalienceCmap();

    const rowsByClusters: TableData[] =
        topTokenInfosByClusters.map((topTokenInfos, clusterIdx) => {
          const tokensDom = topTokenInfos.topTokenInfos.map(
              tokenInfo => this.renderToken(
                  tokenInfo.token, tokenInfo.weight, unsignedCmap, gradKey));
          return [clusterIdx, html`${tokensDom}`];
        });

    const onSelectClusters = (clusterIdxs: number[]) => {
      const dataPointIds: string[] = clusterInfos
        .filter(clusterInfo => clusterIdxs.includes(clusterInfo.clusterId))
        .map(clusterInfo => clusterInfo.exampleId);
      this.selectionService.selectIds(dataPointIds, this);
    };

    // clang-format off
    return html`
      <div class="grad-key-row">
        <div class="grad-key-label">${gradKey}</div>
        <lit-data-table
          .columnNames=${['Cluster Index', 'Top Tokens']}
          .data=${rowsByClusters}
          selectionEnabled
          .onSelect=${onSelectClusters}
        ></lit-data-table>
      </div>`;
    // clang-format on
  }

  renderTopTokens() {
    const {topTokenInfosByClusters, clusterInfos} = this.state.clusteringState;
    const gradKeys = Object.keys(topTokenInfosByClusters);
    // clang-format off
    return html`
      ${gradKeys.map(
          gradKey => this.renderSingleGradKeyTopTokenInfos(
            gradKey, topTokenInfosByClusters[gradKey], clusterInfos[gradKey]))}
      ${this.state.clusteringState.isLoading ? this.renderSpinner() : null}`;
    // clang-format on
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
      // clang-format off
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
      // clang-format on
    };
    // Always show the clustering config first.
    const interpreters: string[] = [
      SALIENCE_CLUSTERING_INTERPRETER_NAME,
      ...this.getSalienceInterpreterNames()
    ];

    // clang-format off
    return html`
      <div class="controls-and-results-container">
        <div class="clustering-controls-container">
          ${interpreters.map(renderInterpreterControls)}
        </div>
        <div class="clustering-results-container">
          ${this.renderTopTokens()}
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
      <span class="selection-warning">
        Please select no datapoint (for entire dataset) or >= 2 datapoints to
        compute clusters.
      </span>`;
    // clang-format on
  }

  private renderColorLegend() {
    const colorMap = new UnsignedSalienceCmap();

    // TODO(b/263270935): Use a toColorLegend method to avoid D3-like style.
    function scale(val: number) {
      return colorMap.bgCmap(val);
    }
    scale.domain = () => colorMap.colorScale.domain();

    // clang-format off
    return html`
        <div class="color-legend-container">
          <color-legend legendType=${LegendType.SEQUENTIAL}
            .scale=${scale}
            numBlocks=5}>
          </color-legend>
        </div>`;
    // clang-format on
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
            ${this.statusMessage}
          </p>
          ${this.renderColorLegend()}
        </div>
      </div>`;
    // clang-format on
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
