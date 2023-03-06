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

import '../elements/showmore';
import {html, TemplateResult} from 'lit';
// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {InterpreterClick} from '../elements/interpreter_controls';
import {TableData} from '../elements/table';
import {TokenWithWeight} from '../elements/token_chips';
import {CategoryLabel, FieldMatcher, TokenSalience} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, IndexedInput, ModelInfoMap} from '../lib/types';
import {cloneSpec, createLitType, findSpecKeys} from '../lib/utils';
import {SalienceCmap, SignedSalienceCmap, UnsignedSalienceCmap} from '../services/color_service';
import {ColumnData} from '../services/data_service';
import {DataService} from '../services/services';

import {styles} from './salience_clustering_module.css';

const SALIENCE_MAPPER_KEY = 'Salience Mapper';
const N_CLUSTERS_KEY = 'Number of clusters';
const TOP_K_TOKENS_KEY =
    'Number of top salient tokens to consider per data point';
const N_TOKENS_TO_DISPLAY_KEY = 'Number of tokens to display per cluster';
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
  topTokenVisible: {[gradKey: string]:boolean[]};
  isLoading: boolean;
  colorMap: {[gradKey: string]: SalienceCmap};
  clusteringConfig: {[gradKey: string]: CallConfig};
}

// Clustering assignment result for each data point.
interface ClusterInfo {
  exampleId: string;
  clusterId: number;
}

// Top token information per cluster, sorted by the ascending by cluster IDs.
interface TopTokenInfosByCluster {
  topTokenInfos: TokenWithWeight[];
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

  // For color legend
  private readonly cmapGamma: number = 2.0;
  private readonly signedCmap = new SignedSalienceCmap(this.cmapGamma);
  private readonly unsignedCmap = new UnsignedSalienceCmap(this.cmapGamma);

  // Mapping from salience mapper to clustering results.
  @observable
  private state: ModuleState = {
    dataColumns: [],
    clusteringConfig: {},
    salienceConfigs: {},
    clusteringState: {
      clusterInfos: {},
      topTokenInfosByClusters: {},
      isLoading: false,
      topTokenVisible: {},
      colorMap: {},
      clusteringConfig: {},
    },
  };

  private expandDisabled = false;
  private collapseDisabled = true;

  override firstUpdated() {
    const state: ModuleState = {
      dataColumns: [],
      clusteringConfig: {},
      salienceConfigs: {},
      clusteringState: {
        clusterInfos: {},
        topTokenInfosByClusters: {},
        isLoading: false,
        topTokenVisible: {},
        colorMap: {},
        clusteringConfig: {},
      },
    };

    this.state = state;
  }

  private async runInterpreter() {
    const salienceMapper = this.state.clusteringConfig[SALIENCE_MAPPER_KEY];
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
      const visible = [];
      // Store per-example cluster IDs.
      const clusterInfos: ClusterInfo[] = [];
      const clusterIds = clusteringResult['cluster_ids'][gradKey];
      const featName = 'saliency cluster index (' +
          `${this.state.clusteringConfig[SALIENCE_MAPPER_KEY]}, ${gradKey}, ` +
          `${this.state.clusteringConfig[N_CLUSTERS_KEY]}, ` +
          `${this.state.clusteringConfig[TOP_K_TOKENS_KEY]}, ` +
          `${this.runCount})`;
      const dataMap: ColumnData = new Map();
      this.state.clusteringState.clusteringConfig[gradKey] = {
          ...this.state.clusteringConfig};

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

        visible.push(false);

        for (const topTokenTuple of topTokenInfosByClusters[clusterId]) {
          topTokenInfosByCluster.topTokenInfos.push(
              {token: topTokenTuple[0], weight: topTokenTuple[1]});
        }
        this.state.clusteringState.topTokenInfosByClusters[gradKey].push(
            topTokenInfosByCluster);

      this.state.clusteringState.topTokenVisible[gradKey] = visible;
      }

      // Determine color map.
      const interpreters = this.appState.metadata.interpreters;
      const salienceSpecInfo =
          interpreters[salienceMapper].metaSpec['saliency'] as TokenSalience;
      this.state.clusteringState.colorMap[gradKey] =
          !!salienceSpecInfo.signed ? this.signedCmap : this.unsignedCmap;
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

  // Render a table that lists clusters with their top tokens.
  private renderSingleGradKeyTopTokenInfos(
      gradKey: string, topTokenInfosByClusters: TopTokenInfosByCluster[],
      clusterInfos: ClusterInfo[], colorMap: SalienceCmap) {
    // TODO(b/268221058): Compute number of data points per cluster from backend
    // clang-format off
    const rowsByClusters: TableData[] =
      topTokenInfosByClusters.map((topTokenInfos, clusterIdx) => {

        const defaultLength = 5;

        const onChange = (e:Event) => {
          this.state.clusteringState.topTokenVisible[gradKey][clusterIdx] = true;
          this.collapseDisabled = false;
        };

        const renderedTokens =
          this.state.clusteringState.topTokenVisible[gradKey][clusterIdx] ?
          topTokenInfos.topTokenInfos :
          topTokenInfos.topTokenInfos.slice(0,
            Math.min(defaultLength, topTokenInfos.topTokenInfos.length));

        const renderShowMore =
          renderedTokens.length === topTokenInfos.topTokenInfos.length ?
          "" :
          html`<lit-showmore
            @show-more=${onChange}>
            </lit-showmore>`;

        return [
          clusterIdx,
          clusterInfos.filter(
            clusterInfo => clusterInfo.clusterId === clusterIdx).length,
          html`
          <lit-token-chips
            .tokensWithWeights=${renderedTokens}
            .cmap=${colorMap}>
          </lit-token-chips>
          ${renderShowMore}`
        ];
      });
    // clang-format on

    const onSelectClusters = (clusterIdxs: number[]) => {
      const dataPointIds: string[] = clusterInfos
        .filter(clusterInfo => clusterIdxs.includes(clusterInfo.clusterId))
        .map(clusterInfo => clusterInfo.exampleId);
      this.selectionService.selectIds(dataPointIds, this);
    };

    const numTokensPerCluster =
        this.state.clusteringState
            .clusteringConfig[gradKey][N_TOKENS_TO_DISPLAY_KEY];

    // clang-format off
    return html`
      <div class="grad-key-row">
        <div class="grad-key-label">${gradKey}</div>
        <lit-data-table
          .columnNames=${[
            'Cluster index',
            'N',
            `Tokens with high average saliency (up to ${numTokensPerCluster})`
          ]}
          .data=${rowsByClusters}
          selectionEnabled
          .onSelect=${onSelectClusters}
        ></lit-data-table>
      </div>`;
    // clang-format on
  }

  renderTopTokens() {
    const {topTokenInfosByClusters, clusterInfos, colorMap} =
        this.state.clusteringState;
    const gradKeys = Object.keys(topTokenInfosByClusters);
    // clang-format off
    return html`
      ${gradKeys.map(
          gradKey => this.renderSingleGradKeyTopTokenInfos(
            gradKey, topTokenInfosByClusters[gradKey], clusterInfos[gradKey],
            colorMap[gradKey]))}
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

  private renderColorLegend(
      colorName: string, colorMap: SalienceCmap, numBlocks: number) {
    // TODO(b/263270935): Use a toColorLegend method to avoid D3-like style.
    function scale(val: number) {
      return colorMap.bgCmap(val);
    }
    scale.domain = () => colorMap.colorScale.domain();

    // clang-format off
    return html`
        <color-legend legendType=${LegendType.SEQUENTIAL}
          label=${colorName}
          .scale=${scale}
          numBlocks=${numBlocks}>
        </color-legend>`;
    // clang-format on
  }

  private renderColorLegends() {
    // Determine which color maps are currently used.
    const enabledColorMaps: {[key: string]: TemplateResult} = {};
    for (const gradKey of Object.keys(this.state.clusteringState.colorMap)) {
      if (this.state.clusteringState.colorMap[gradKey] === this.signedCmap) {
        enabledColorMaps['Signed'] =
            this.renderColorLegend('Signed', this.signedCmap, 7);
      }
      if (this.state.clusteringState.colorMap[gradKey] === this.unsignedCmap) {
        enabledColorMaps['Unsigned'] =
            this.renderColorLegend('Unsigned', this.unsignedCmap, 5);
      }
    }
    return Object.values(enabledColorMaps);
  }

  override renderImpl() {
    const updateVisible = (visible: boolean) => {
      for (const gradKey of Object.keys(this.state.clusteringState.topTokenVisible)) {
        const numClusters = this.state.clusteringState.topTokenVisible[gradKey].length;
        for (let i = 0; i < numClusters; i++) {
          this.state.clusteringState.topTokenVisible[gradKey][i] = visible;
        }
      }
    };

    const onClickExpand = () => {
      updateVisible(true);
      this.expandDisabled = true;
      this.collapseDisabled = false;
    };

    const onClickCollapse = () => {
      updateVisible(false);
      this.collapseDisabled = true;
      this.expandDisabled = false;

    };
    // clang-format off
    return html`
      <div class="module-container">
        <div class="module-results-area">
          ${this.renderControlsAndResults()}
        </div>
        <div class="module-footer">
          <p class="module-status">
            ${this.canRunClustering ? '' : this.renderSelectionWarning()}
            ${this.statusMessage}
          </p>
          <div class="color-legend-container">
            ${this.renderColorLegends()}
          </div>
          <div id="toolbar-buttons">
            <button class='hairline-button' @click=${onClickCollapse} ?disabled="${this.collapseDisabled}">
              Collapse all
            </button>
            <button class='hairline-button' @click=${onClickExpand} ?disabled="${this.expandDisabled}">
              Expand all
            </button>
          </div>
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
