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

import '../elements/line_chart';
import '../elements/bar_chart';

import {html} from 'lit';
// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators';
import {observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, Input, ModelInfoMap, Spec} from '../lib/types';

import {styles} from './salience_clustering_module.css';

interface SalienceMapperToClusteringState {
  [salienceMapper: string]: ClusteringState;
}

interface ClusteringState {
  dataColumns: string[];
  clusterInfos: {[name: string]: ClusterInfo[]};
  isLoading: boolean;
  config?: CallConfig;
}

// Clustering result for a single piece of text.
interface ClusterInfo {
  example: Input;
  clusterId: number;
}

/**
 * A LIT module that renders salience clustering results.
 */
@customElement('salience-clustering-module')
export class SalienceClusteringModule extends LitModule {
  static override title = 'Salience Clustering Results';
  static override numCols = 1;
  // TODO(b/215497716): Get the salience mappers from the interpreter component
  // and let the user select the wanted one.
  private readonly salienceMapper = 'Grad L2 Norm';
  static override template = (model = '', selectionServiceIndex = 0) => {
    // clang-format off
    return html`
        <salience-clustering-module
            model=${model} selectionServiceIndex=${selectionServiceIndex}>
        </salience-clustering-module>`;
    // clang format on
  };

  static override get styles() {
    return [sharedStyles, styles];
  }

  // Mapping from salience mapper to clustering results.
  @observable private state: SalienceMapperToClusteringState = {};

  override firstUpdated() {
    const state: SalienceMapperToClusteringState = {};
    state[this.salienceMapper] = {
      dataColumns: [],
      clusterInfos: {},
      isLoading: false,
      // TODO(b/216772288): Load config key from interpreter.
      config: {'Salience Mapper': this.salienceMapper},
    };
    this.state = state;
  }

  private runInterpreterDefault() {
    return this.runInterpreter(this.salienceMapper);
  }

  private async runInterpreter(salienceMapper: string) {
    this.state[salienceMapper].clusterInfos = {};
    const input = this.selectionService.selectedOrAllInputData;
    if (!this.canRunClustering) {
      return;
    }

    this.state[salienceMapper].isLoading = true;
    const promise = this.apiService.getInterpretations(
        input, this.model, this.appState.currentDataset, 'salience clustering',
        this.state[salienceMapper].config, `Running ${salienceMapper}`);
    const clusteringResult =
        await this.loadLatest(`interpretations-${salienceMapper}`, promise);
    this.state[salienceMapper].isLoading = false;
    this.state[salienceMapper].dataColumns = Object.keys(input[0].data);

    for (const gradKey of Object.keys(clusteringResult['cluster_ids'])) {
      const clusterInfos: ClusterInfo[] = [];
      const clusterIds = clusteringResult['cluster_ids'][gradKey];

      for (let i = 0; i < clusterIds.length; i++) {
        const clusterInfo: ClusterInfo = {
            example: input[i].data, clusterId: clusterIds[i]};
        clusterInfos.push(clusterInfo);
      }
      this.state[salienceMapper].clusterInfos[gradKey] = clusterInfos;
    }
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

  renderClusteringState(salienceMapper: string) {
    const renderSingleGradKeyClusterInfos =
        (gradKey: string, clusterInfos: ClusterInfo[]) => {
          const rows: TableData[] = clusterInfos.map((clusterInfo) => {
            const row: string[] = [];
            for (const dataColumn of this.state[salienceMapper].dataColumns) {
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
                    [...this.state[salienceMapper].dataColumns, 'cluster ID']}
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
              this.state[salienceMapper].isLoading ? this.renderSpinner() :
                                                     null}`;
          // clang format on
        };

    return renderClusterInfos(this.state[salienceMapper].clusterInfos);
  }

  renderTable() {
    if (this.state[this.salienceMapper] == null ||
        this.state[this.salienceMapper].clusterInfos == null) {
      return html`<div>Nothing to show.</div>`;
    }
    // clang-format off
    return html`
      <div class="clustering-salience-mapper">${this.salienceMapper}</div>
        ${this.renderClusteringState(this.salienceMapper)}`;
    // clang format on
  }

  private get canRunClustering() {
    const input = this.selectionService.selectedOrAllInputData;
    return (input != null && input.length >= 2);
  }

  renderSelectionWarning() {
    if (this.canRunClustering) {
      return html``;
    } else {
      // clang-format off
      return html`
        <div class="selection-warning">
          Please select no datapoint (for entire dataset) or >= 2 datapoints to
          compute clusters.
        </div>`;
      // clang format on
    }
  }

  renderControls() {
    // clang-format off
    return html`
      <button class='hairline-button'
        ?disabled="${!this.canRunClustering}"
        @click=${this.runInterpreterDefault}>
        Compute clusters
      </button>
      ${this.renderSelectionWarning()}`;
    // clang format on
  }

  override render() {
    // clang-format off
    return html`
      <div class='module-container'>
        <div class="module-toolbar">
          ${this.renderControls()}
        </div>
        <div class='module-results-area'>
          ${this.renderTable()}
        </div>
      </div>`;
    // clang format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap,
                                      datasetSpec: Spec) {
    for (const modelInfo of Object.values(modelSpecs)) {
      if (modelInfo.interpreters.indexOf('salience clustering') !== -1) {
        return true;
      }
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'salience-clustering-module': SalienceClusteringModule;
  }
}
