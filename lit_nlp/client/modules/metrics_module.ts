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
import {customElement, query} from 'lit/decorators';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {FacetsChange} from '../core/faceting_control';
import {LitModule} from '../core/lit_module';
import {ColumnHeader, DataTable, TableData} from '../elements/table';
import {MetricBestValue, MetricResult} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, FacetMap, IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {GroupService, NumericFeatureBins} from '../services/group_service';
import {ClassificationService, SliceService} from '../services/services';

// Each entry from the server.
interface MetricsResponse {
  // Using case to achieve parity with the property names in Python code
  // tslint:disable-next-line:enforce-name-casing
  pred_key: string;
  // tslint:disable-next-line:enforce-name-casing
  label_key: string;
  metrics: MetricsValues;
}

// A dict of metrics type to the MetricsValues for one metric generator.
interface ModelHeadMetrics {
  [metricsType: string]: MetricsValues;
}

// A dict of metric names to values, from one metric generator.
interface MetricsValues {
  [metricName: string]: number;
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
  @observable private facetBySlice: boolean = false;
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

  override firstUpdated() {
    this.react(() => this.appState.currentInputData, entireDataset => {
      this.metricsMap = {};
      this.addMetrics(this.appState.currentInputData, Source.DATASET);
      this.updateAllFacetedMetrics();
    });
    this.reactImmediately(() => this.selectionService.selectedInputData, () => {
      // When the selection changes, remove all existing selection-based rows
      // from the metrics table.
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
        // If selection changed outside of this module, clear the highlight in
        // the metrics table.
        this.table.primarySelectedIndex = -1;
        this.table.selectedIndices = [];
      }
      if (this.selectionService.selectedInputData.length > 0) {
        // If a selection is made outside of this module,, then calculate a row
        // in the metrics table for the selection.
        this.addMetrics(
            this.selectionService.selectedInputData, Source.SELECTION);
      }
    });
    this.react(() => this.classificationService.allMarginSettings, margins => {
      this.addMetrics(this.appState.currentInputData, Source.DATASET);
      this.updateAllFacetedMetrics();
    });
    this.react(() => this.sliceService.sliceNames, slices => {
      this.facetBySlice = true;
      this.updateSliceMetrics();
    });

    // Do this once, manually, to avoid duplicate calls on load.
    this.addMetrics(this.appState.currentInputData, Source.DATASET);
    this.updateAllFacetedMetrics();
  }

  /** Gets and adds metrics information for datapoints to the metricsMap. */
  async addMetrics(datapoints: IndexedInput[], source: Source,
                   facetMap?: FacetMap, displayName?: string) {
    const models = this.appState.currentModels;

    // Get the metrics for all models for the provided datapoints.
    const datasetMetrics = await Promise.all(models.map(
        async (model: string) => this.getMetrics(datapoints, model)));

    let name = displayName != null ? displayName : source.toString();
    if (facetMap !=null) {
      name += ' (faceted)';
    }

    // Add the returned metrics for each model and head to the metricsMap.
    datasetMetrics.forEach((returnedMetrics, i) => {
      Object.keys(returnedMetrics).forEach(metricsType => {
        const metricsRespones: MetricsResponse[] = returnedMetrics[metricsType];
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

  private async getMetrics(selectedInputs: IndexedInput[], model: string) {
    this.pendingCalls += 1;
    try {
      const config =
          this.classificationService.marginSettings[model] as CallConfig || {};
      const metrics = await this.apiService.getInterpretations(
          selectedInputs, model, this.appState.currentDataset, 'metrics', config);
      this.pendingCalls -= 1;
      return metrics;
    }
    catch {
      this.pendingCalls -= 1;
      return {};
    }
  }

  /** Convert the metricsMap information into table data for display. */
  @computed
  get tableData(): TableHeaderAndData {
    const {metaSpec} = this.appState.metadata.interpreters['metrics'];
    if (metaSpec == null) return {'header': [], 'data': []};

    const tableRows: TableData[] = [];
    const metricBests = new Map<string, number>();
    function getMetricKey(t: string, n: string) {return `${t}: ${n}`;}

    Object.values(this.metricsMap).forEach(row => {
      Object.entries(row.headMetrics).forEach(([metricsT, metricsV]) => {
        Object.entries(metricsV).forEach(([name, val]) => {
          const key = getMetricKey(metricsT, name);
          const max = metricBests.get(key)!;
          const spec = metaSpec[key];
          if (!(spec instanceof MetricResult)) return;
          const bestCase = spec.best_value;

          if (bestCase != null && (!metricBests.has(key) ||
              (bestCase === MetricBestValue.HIGHEST && max < val) ||
              (bestCase === MetricBestValue.LOWEST && max > val) ||
              (bestCase === MetricBestValue.ZERO && Math.abs(max) > Math.abs(val)))) {
            metricBests.set(key,
                            bestCase === MetricBestValue.NONE ? Infinity : val);
          }
        });
      });
    });

    const metricNames = [...metricBests.keys()];

    for (const row of Object.values(this.metricsMap)) {
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
      const spec = metaSpec[name] as MetricResult;
      return {name, rightAlign: true, html: html`
        <div class="header-text" title=${spec.description}>${name}</div>`
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
    for (const modelInfo of Object.values(modelSpecs)) {
      if (modelInfo.interpreters.indexOf('metrics') !== -1) {
        return true;
      }
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'metrics-module': MetricsModule;
  }
}
