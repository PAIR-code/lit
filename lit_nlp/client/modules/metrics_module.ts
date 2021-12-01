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

import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {CallConfig, FacetMap, IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {GroupService, FacetingMethod, FacetingConfig, NumericFeatureBins} from '../services/group_service';
import {ClassificationService, SliceService} from '../services/services';

import {styles} from './metrics_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

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
  header: string[];
  data: TableData[];
}

/**
 * Module to show metrics of a model.
 */
@customElement('metrics-module')
export class MetricsModule extends LitModule {
  static override title = 'Metrics';
  static override numCols = 6;
  static override template = () => {
    return html`<metrics-module></metrics-module>`;
  };
  static override duplicateForModelComparison = false;

  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly sliceService = app.getService(SliceService);
  private readonly groupService = app.getService(GroupService);
  private readonly classificationService =
      app.getService(ClassificationService);

  @observable private selectedFacetBins: NumericFeatureBins = {};

  @observable private metricsMap: MetricsMap = {};
  @observable private facetBySlice: boolean = false;
  @observable private selectedFacets: string[] = [];
  @observable private pendingCalls = 0;


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
        return;
      }
      if (this.selectionService.selectedInputData.length > 0) {
        this.addMetrics(this.selectionService.selectedInputData,
                        Source.SELECTION);
        this.updateFacetedMetrics(this.selectionService.selectedInputData,
                                  true);
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
      this.updateFacetedMetrics(this.selectionService.selectedInputData, true);
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
    const tableRows = [] as TableData[];
    const allMetricNames = new Set<string>();
    Object.values(this.metricsMap).forEach(row => {
      Object.keys(row.headMetrics).forEach(metricsType => {
        const metricsValues = row.headMetrics[metricsType];
        Object.keys(metricsValues).forEach(metricName => {
          allMetricNames.add(`${metricsType}: ${metricName}`);
        });
      });
    });

    const metricNames = [...allMetricNames];

    for (const row of Object.values(this.metricsMap)) {
      const rowMetrics = metricNames.map(metricKey => {
        const [metricsType, metricName] = metricKey.split(": ");
        if (row.headMetrics[metricsType] == null) {
          return '-';
        }
        const num = row.headMetrics[metricsType][metricName];
        if (num == null) {
          return '-';
        }
        // If the metric is not a whole number, then round to 3 decimal places.
        if (typeof num === 'number' && num % 1 !== 0) {
          return num.toFixed(3);
        }
        return num;
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

    return {
      'header': [
        'Model', 'From', ...this.selectedFacets, 'Field', 'N', ...metricNames
      ],
      'data': tableRows
    };
  }

  override render() {
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
    // TODO(b/180903904): Add onSelect behavior to rows for selection.
    return html`
      <lit-data-table
        .columnNames=${this.tableData.header}
        .data=${this.tableData.data}
      ></lit-data-table>
    `;
  }

  renderFacetSelector() {
    // Update the filterdict to match the checkboxes.
    const onFeatureCheckboxChange = (e: Event, key: string) => {
      if ((e.target as HTMLInputElement).checked) {
        this.selectedFacets.push(key);
      } else {
        const index = this.selectedFacets.indexOf(key);
        this.selectedFacets.splice(index, 1);
      }

      const configs: FacetingConfig[] = this.selectedFacets.map(feature => ({
          featureName: feature,
          method: this.groupService.numericalFeatureNames.includes(feature) ?
                  FacetingMethod.EQUAL_INTERVAL : FacetingMethod.DISCRETE
      }));

      this.selectedFacetBins = this.groupService.numericalFeatureBins(configs);
      this.updateAllFacetedMetrics();
    };

    // Disable the "slices" on the dropdown if all the slices are empty.
    const slicesDisabled = this.sliceService.areAllSlicesEmpty();

    const onSlicesCheckboxChecked = (e: Event) => {
      this.facetBySlice = !this.facetBySlice;
      this.updateSliceMetrics();
    };
    // clang-format off
    return html`
      <label class="cb-label">Show slices</label>
      <lit-checkbox
        ?checked=${this.facetBySlice}
        @change=${onSlicesCheckboxChecked}
        ?disabled=${slicesDisabled}>
      </lit-checkbox>
      <label class="cb-label">Facet by</label>
       ${
        this.groupService.denseFeatureNames.map(
            facetName => this.renderCheckbox(facetName, false,
                (e: Event) => {onFeatureCheckboxChange(e, facetName);}, false))}
      ${this.pendingCalls > 0 ? this.renderSpinner() : null}
    `;
    // clang-format on
  }

  private renderCheckbox(
      key: string, checked: boolean, onChange: (e: Event, key: string) => void,
      disabled: boolean) {
    // clang-format off
    return html`
        <div class='checkbox-holder'>
          <lit-checkbox
            ?checked=${checked}
            ?disabled=${disabled}
            @change='${(e: Event) => {onChange(e, key);}}'
            label=${key}>
          </lit-checkbox>
        </div>
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
