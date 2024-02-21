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

import {html} from 'lit';
import {customElement, query} from 'lit/decorators.js';
import {styleMap} from 'lit/directives/style-map.js';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {FacetsChange} from '../core/faceting_control';
import {LitModule} from '../core/lit_module';
import {ColumnHeader, DataTable, TableData} from '../elements/table';
import {MetricBestValue, MetricResult} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, FacetMap, IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {MetricsResponse, MetricsValues} from '../services/api_service';
import {GroupService, type NumericFeatureBins} from '../services/group_service';
import {ClassificationService, SliceService} from '../services/services';

// A dict of metrics type to the MetricsValues for one metric generator.
interface ModelHeadMetrics {
  [metricsType: string]: MetricsValues;
}

// The source of datapoints for a row in the metrics table.
enum Source {
  DATASET = "dataset",
  SELECTION = "selection",
  SLICE = "slice"
}

// Data for rendering a row in the table.
interface MetricsRow {
  model: string;
  selection: string;
  predKey: string;
  exampleIds: string[];
  headMetrics: ModelHeadMetrics;
  source: Source;
  facets?: FacetMap;
}

// A dict of row keys to metrics row information, to store all metric info
// for the metrics table.
interface MetricsMap {
  [rowKey: string]: MetricsRow;
}

// Data to render the metrics table, created from the MetricsMap.
interface TableHeaderAndData {
  header: Array<string|ColumnHeader>;
  data: TableData[];
}

/**
 * Module to show metrics of a model.
 */
@customElement('metrics-module')
export class MetricsModule extends LitModule {
  static override title = 'Metrics';
  static override numCols = 6;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`
  <metrics-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </metrics-module>`;
  static override duplicateForModelComparison = false;

  static override get styles() {
    return [sharedStyles];
  }

  private readonly sliceService = app.getService(SliceService);
  private readonly groupService = app.getService(GroupService);
  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly facetingControl = document.createElement('faceting-control');

  @observable private selectedFacetBins: NumericFeatureBins = {};

  @observable private metricsMap: MetricsMap = {};
  @observable private facetBySlice = false;
  @observable private selectedFacets: string[] = [];
  @observable private pendingCalls = 0;
  @query('#metrics-table') private readonly table?: DataTable;

  constructor() {
    super();

    const facetsChange = (event: CustomEvent<FacetsChange>) => {
      this.selectedFacets = event.detail.features;
      this.selectedFacetBins = event.detail.bins;
      this.updateAllFacetedMetrics();
    };

    this.facetingControl.contextName = MetricsModule.title;
    this.facetingControl.addEventListener(
        'facets-change', facetsChange as EventListener);
  }

  override connectedCallback() {
    super.connectedCallback();

    this.react(
        () => this.appState.currentInputData,
        () => {
          this.metricsMap = {};
          this.addMetrics(this.appState.currentInputData, Source.DATASET);
          this.updateAllFacetedMetrics();
        });
    this.react(
        () => this.classificationService.allMarginSettings,
        () => {
          this.addMetrics(this.appState.currentInputData, Source.DATASET);
          this.addMetrics(
            this.selectionService.selectedInputData, Source.SELECTION);
          this.updateAllFacetedMetrics();
        });
    this.react(
        () => this.sliceService.sliceNames,
        () => {
          this.facetBySlice = true;
          this.updateSliceMetrics();
        });

    this.react(
        () => this.selectionService.selectedInputData,
        () => {
          // When the selection changes, remove all existing selection-based
          // rows from the metrics table.
          Object.keys(this.metricsMap).forEach(key => {
            if (this.metricsMap[key].source === Source.SELECTION) {
              delete this.metricsMap[key];
            }
          });
          if (this.selectionService.lastUser === this) {
            // If selection made through this module, no need to show a separate
            // "selection" row in the metrics table, as the selected row will
            // be highlighted to indicate that it is selected.
            return;
          } else if (this.table != null) {
            // If selection changed outside of this module, clear the highlight
            // in the metrics table.
            this.table.primarySelectedIndex = -1;
            this.table.selectedIndices = [];
          }
          if (this.selectionService.selectedInputData.length > 0) {
            // If a selection is made outside of this module,, then calculate a
            // row in the metrics table for the selection.
            this.addMetrics(
                this.selectionService.selectedInputData, Source.SELECTION);
          }
        });

    // Do this once, manually, to avoid duplicate calls on load.
    this.addMetrics(this.appState.currentInputData, Source.DATASET);
    this.addMetrics(
      this.selectionService.selectedInputData, Source.SELECTION);
    this.updateAllFacetedMetrics();
  }

  /** Gets and adds metrics information for datapoints to the metricsMap. */
  async addMetrics(datapoints: IndexedInput[], source: Source,
                   facetMap?: FacetMap, displayName?: string) {
    const {currentModels: models} = this.appState;

    // Get the metrics for all models for the provided datapoints.
    const datasetMetrics = await Promise.all(
        models.map(async (model: string) => this.getMetrics(datapoints, model))
    );

    let name = displayName != null ? displayName : source.toString();
    if (facetMap !=null) {
      name += ' (faceted)';
    }

    // Add the returned metrics for each model and head to the metricsMap.
    datasetMetrics.forEach((returnedMetrics, i) => {
      Object.entries(returnedMetrics).forEach(([metricsType, metricsRespones]) => {
        metricsRespones.forEach(metricsResponse => {
          const rowKey = this.getRowKey(
              models[i], name, metricsResponse.pred_key, facetMap);
          if (this.metricsMap[rowKey] == null) {
            this.metricsMap[rowKey] = {
              model: models[i],
              selection: name,
              exampleIds: datapoints.map(datapoint => datapoint.id),
              predKey: metricsResponse.pred_key,
              headMetrics: {},
              facets: facetMap,
              source
            };
          }
          this.metricsMap[rowKey].exampleIds = datapoints.map(
              datapoint => datapoint.id);

          // Each model/datapoints/head combination stores a dict of metrics
          // for the different metrics generators run by LIT.
          this.metricsMap[rowKey].headMetrics[metricsType] =
              metricsResponse.metrics;
        });
      });
    });
  }

  /** Returns a MetricsRow key based on arguments. */
  getRowKey(model: string, datapointsId: string, predKey: string,
            facetMap?: FacetMap) {
    let facetString = '';
    if (facetMap != null) {
      for (const facetVal of Object.values(facetMap)) {
        facetString += `${facetVal.displayVal}-`;
      }
    }
    return `${model}-${datapointsId}-${predKey}-${facetString}`;
  }

  private updateFacetedMetrics(datapoints: IndexedInput[],
                               isSelection: boolean ) {
    // Get the intersectional feature bins.
    if (this.selectedFacets.length > 0) {
      const groupedExamples = this.groupService.groupExamplesByFeatures(
          this.selectedFacetBins,
          datapoints,
          this.selectedFacets);

      const source =  isSelection ? Source.SELECTION : Source.DATASET;
      // Manually set all of their display names.
      Object.keys(groupedExamples).forEach(key => {
        this.addMetrics(groupedExamples[key].data, source,
                        groupedExamples[key].facets);
      });
    }
  }

  private updateAllFacetedMetrics() {
    Object.keys(this.metricsMap).forEach(key => {
      if (this.metricsMap[key].facets != null) {
        delete this.metricsMap[key];
      }
    });
    // Get the intersectional feature bins.
    if (this.selectedFacets.length > 0) {
      this.updateFacetedMetrics(this.appState.currentInputData, false);
    }
  }

  /**
   * Facet the data by slices.
   */
  private updateSliceMetrics() {
    Object.keys(this.metricsMap).forEach(key => {
      if (this.metricsMap[key].source === Source.SLICE) {
        delete this.metricsMap[key];
      }
    });
    if (this.facetBySlice) {
      this.sliceService.sliceNames.forEach(name => {
        const data = this.sliceService.getSliceDataByName(name);
        if (data.length > 0) {
          this.addMetrics(data, Source.SLICE, /* facetMap */ undefined, name);
        }
      });
    }
  }

  private async getMetrics(
      selectedInputs: IndexedInput[], model: string): Promise<MetricsResponse> {
    const {currentDataset} = this.appState;
    const {
      metrics: compatMetrics,
      datasets: compatDatasets
    } = this.appState.metadata.models[model];

    if (!compatDatasets.includes(currentDataset)) {
      // There is a small window in which the this.appState.currentModels has
      // updated for a render loop but the this.appState.currentDataset has not,
      // after changing the relevant selections in the Global Settings. This
      // guard prevents this module from requesting metrics computations for
      // incompatible model/dataset pairs.
      // TODO(b/297072974): Console logging the current model and dataset from
      // this line, and checking this against requests in the Network tab of
      // Chrome DevTools, is a good place to verify the possible timing bug.
      return Promise.resolve({});
    }

    this.pendingCalls += 1;
    // TODO(b/254832560): Allow the user to configure which metrics component
    // are run via the UI and pass them in to this ApiService call.
    const metricsToRun = compatMetrics.length ? compatMetrics.join(',') : '';
    const config =
        this.classificationService.marginSettings[model] as CallConfig || {};

    let metrics: MetricsResponse;
    if (selectedInputs.length) {
      try {
        metrics = await this.apiService.getMetrics(
            selectedInputs, model, currentDataset, metricsToRun, config);
      } catch {
        metrics = {};
      }
    } else {
      metrics = {};
    }

    this.pendingCalls -= 1;
    return metrics;
  }

  /** Convert the metricsMap information into table data for display. */
  @computed get tableData(): TableHeaderAndData {
    const {currentModels} = this.appState;
    const metricsInfo = this.appState.metadata.metrics;
    if (Object.keys(metricsInfo).length === 0) {
      return {'header': [], 'data': []};
    }

    const tableRows: TableData[] = [];
    const metricBests = new Map<string, number>();
    function getMetricKey(t: string, n: string) {return `${t}: ${n}`;}

    Object.values(this.metricsMap).forEach(row => {
      Object.entries(row.headMetrics).forEach(([metricsT, metricsV]) => {
        const {metaSpec} = metricsInfo[metricsT];
        Object.entries(metricsV).forEach(([name, val]) => {
          const key = getMetricKey(metricsT, name);
          const max = metricBests.get(key)!;
          const spec = metaSpec[name];
          if (!(spec instanceof MetricResult)) return;
          const bestCase = spec.best_value;

          const bestsUndefined = !metricBests.has(key);
          const bestIshighValIsHigher =
              bestCase === MetricBestValue.HIGHEST && max < val;
          const bestIsLowValIsLower =
              bestCase === MetricBestValue.LOWEST && max > val;
          const bestIsZeroValIsCloser =
              bestCase === MetricBestValue.ZERO && Math.abs(max) > Math.abs(val);
          const shouldUpdate = bestsUndefined || bestIshighValIsHigher ||
                               bestIsLowValIsLower || bestIsZeroValIsCloser;

          if (bestCase != null && shouldUpdate) {
            metricBests.set(key,
                            bestCase === MetricBestValue.NONE ? Infinity : val);
          }
        });
      });
    });

    const metricNames = [...metricBests.keys()];

    const rowsForActiveModels = Object.values(this.metricsMap).filter(
        (row) => currentModels.some((model) => model === row.model));
    for (const row of rowsForActiveModels) {
      const rowMetrics = metricNames.map(metricKey => {
        const [metricsType, metricName] = metricKey.split(": ");
        if (row.headMetrics[metricsType] == null) {return '-';}

        const raw = row.headMetrics[metricsType][metricName];
        if (raw == null) {return '-';}
        const isBest = raw === metricBests.get(metricKey);
        // If the metric is not a whole number, then round to 3 decimal places.
        const value = typeof raw === 'number' && !Number.isInteger(raw) ?
            raw.toFixed(3) : raw;
        const styles = styleMap({'font-weight': isBest ? 'bold' : 'normal'});
        return html`<span style=${styles}>${value}</span>`;
      });
      // Add the "Facet by" columns.
      const rowFacets = this.selectedFacets.map((facet: string) => {
        if (row.facets && row.facets[facet]) {
          return row.facets[facet].displayVal;
        }
        return '-';
      });

      const tableRow = [
        row.model, row.selection, ...rowFacets, row.predKey,
        row.exampleIds.length, ...rowMetrics
      ];
      tableRows.push(tableRow);
    }

    const metricHeaders: ColumnHeader[] = metricNames.map(name => {
      const [metricsType, metricName] = name.split(": ");
      const spec =
          metricsInfo[metricsType].metaSpec[metricName] as MetricResult;
      const [group, metric] = name.split(': ');
      return {
        name,
        html: html`<div slot="tooltip-anchor" class="header-text">
          ${group}<br>${metric}
        </div>`,
        rightAlign: true,
        tooltip: spec.description,
        tooltipWidth: 200,
        width: 100
      };
    });

    return {
      'header': [
        'Model', 'From', ...this.selectedFacets, 'Field', 'N', ...metricHeaders
      ],
      'data': tableRows
    };
  }

  override renderImpl() {
    // clang-format off
    return html`
      <div class="module-container">
        <div class='module-toolbar'>
          ${this.renderFacetSelector()}
        </div>
        <div class='module-results-area'>
          ${this.renderTable()}
        </div>
      </div>
    `;
    // clang-format on
  }

  renderTable() {
    const onSelect = (idxs: number[]) => {
      if (this.table == null) {
        return;
      }
      const primaryId = this.table.primarySelectedIndex;
      if (primaryId < 0) {
        this.selectionService.selectIds([], this);
        this.table.selectedIndices = [];
        return;
      }
      const mapEntry = Object.values(this.metricsMap)[primaryId];
      const ids = mapEntry.exampleIds;
      // If the metrics table row selected isn't the row indicating the current
      // selection, then change the datapoints selection to the ones represented
      // by that row.
      if (mapEntry.source !== Source.SELECTION) {
        this.selectionService.selectIds(ids, this);
        this.table.selectedIndices = [primaryId];
      } else {
        // Don't highlight the row of the selected datapoint if this is clicked
        // as it has no effect.
        this.table.primarySelectedIndex = -1;
        this.table.selectedIndices = [];
      }
    };
    // clang-format off
    return html`
      <lit-data-table id="metrics-table"
        .columnNames=${this.tableData.header}
        .data=${this.tableData.data}
        selectionEnabled
        .onSelect=${(idxs: number[]) => {
          onSelect(idxs);
        }}
      ></lit-data-table>
    `;
    // clang-format on
  }

  renderFacetSelector() {
    // Disable the "slices" on the dropdown if all the slices are empty.
    const slicesDisabled = this.sliceService.areAllSlicesEmpty();

    const onSlicesCheckboxChecked = (e: Event) => {
      this.facetBySlice = !this.facetBySlice;
      this.updateSliceMetrics();
    };

    // clang-format off
    return html`
      <lit-checkbox label="Show slices"
        ?checked=${this.facetBySlice}
        @change=${onSlicesCheckboxChecked}
        ?disabled=${slicesDisabled}>
      </lit-checkbox>
      ${this.facetingControl}
      ${this.pendingCalls > 0 ? this.renderSpinner() : null}
    `;
    // clang-format on
  }

  renderSpinner() {
    return html`
      <lit-spinner size=${24} color="var(--app-secondary-color)">
      </lit-spinner>
    `;
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return Object.values(modelSpecs).some(
        (modelInfo) => modelInfo.metrics.length);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'metrics-module': MetricsModule;
  }
}
