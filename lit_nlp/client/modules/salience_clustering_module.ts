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
import {InterpreterClick} from '../elements/interpreter_controls';
import {TableData} from '../elements/table';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, Input, ModelInfoMap, Spec} from '../lib/types';
import {findSpecKeys, isLitSubtype} from '../lib/utils';

import {styles} from './salience_clustering_module.css';

const SALIENCE_MAPPER_KEY = 'Salience Mapper';
const SALIENCE_CLUSTERING_INTERPRETER_NAME = 'salience clustering';

interface ModuleState {
  dataColumns: string[];
  clusteringConfig: CallConfig;
  salienceConfigs: {[salienceMapper: string]: CallConfig};
  clusteringState: ClusteringState;
}

interface ClusteringState {
  clusterInfos: {[gradKey: string]: ClusterInfo[]};
  isLoading: boolean;
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
  @observable private state: ModuleState = {
      dataColumns: [],
      clusteringConfig: {},
      salienceConfigs: {},
      clusteringState: {
        clusterInfos: {},
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
        isLoading: false,
      },
    };

    this.state = state;
  }

  private async runInterpreter() {
    const salienceMapper = this.state.clusteringConfig[SALIENCE_MAPPER_KEY];
    this.state.clusteringState.clusterInfos = {};
    const input = this.selectionService.selectedOrAllInputData;
    if (!this.canRunClustering) {
      return;
    }
    this.state.clusteringState.isLoading = true;
    const promise = this.apiService.getInterpretations(
        input,
        this.model,
        this.appState.currentDataset,
        SALIENCE_CLUSTERING_INTERPRETER_NAME,
        {...this.state.clusteringConfig,
            ...this.state.salienceConfigs[salienceMapper]},
        `Running ${salienceMapper}`);
    const clusteringResult =
        await this.loadLatest(`interpretations-${salienceMapper}`, promise);
    this.state.clusteringState.isLoading = false;
    this.state.dataColumns = Object.keys(input[0].data);

    for (const gradKey of Object.keys(clusteringResult['cluster_ids'])) {
      const clusterInfos: ClusterInfo[] = [];
      const clusterIds = clusteringResult['cluster_ids'][gradKey];

      for (let i = 0; i < clusterIds.length; i++) {
        const clusterInfo: ClusterInfo = {
            example: input[i].data, clusterId: clusterIds[i]};
        clusterInfos.push(clusterInfo);
      }
      this.state.clusteringState.clusterInfos[gradKey] = clusterInfos;
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

  renderClusteringState() {
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

  private getSalienceInterpreterNames() {
    const names: string[] = [];
    const {interpreters} = this.appState.metadata;
    const validInterpreters =
        this.appState.metadata.models[this.model].interpreters;
    for (const key of validInterpreters) {
      const salienceKeys = findSpecKeys(
          interpreters[key].metaSpec, ['TokenSalience']);
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
      const clonedSpec = JSON.parse(JSON.stringify(spec)) as Spec;
      for (const fieldName of Object.keys(clonedSpec)) {
        // If the generator uses a field matcher, then get the matching
        // field names from the specified spec and use them as the vocab.
        if (isLitSubtype(clonedSpec[fieldName],
                         ['FieldMatcher', 'MultiFieldMatcher'])) {
          clonedSpec[fieldName].vocab =
              this.appState.getSpecKeysFromFieldMatcher(
                  clonedSpec[fieldName], this.model);
        }
      }
      if (Object.keys(clonedSpec).length === 0) {
        return;
      }
      return html`
        <lit-interpreter-controls
          bordered
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
    return html`
      <div class="controls-and-results-container">
        <div class="clustering-controls-container">
          ${interpreters.map(renderInterpreterControls)}
        </div>
        <div class="clustering-results-container">
          ${this.renderClusteringState()}
        </div>
      </div>`;
    // clang-format on
  }

  private get canRunClustering() {
    const input = this.selectionService.selectedOrAllInputData;
    return (input != null && input.length >= 2);
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

  override render() {
    // clang-format off
    return html`
      <div class='module-container'>
        <div class='module-results-area'>
          ${this.renderControlsAndResults()}
        </div>
        ${this.canRunClustering ? html`` : html`
            <div class="module-toolbar footer">
              ${this.renderSelectionWarning()}
            </div>`}
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
