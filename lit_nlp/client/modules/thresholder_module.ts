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

import '../elements/threshold_slider';
// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators';
import {css, html} from 'lit';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';
import {FacetsChange} from '../core/faceting_control';
import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {ColumnHeader, TableEntry} from '../elements/table';
import {ThresholdChange} from '../elements/threshold_slider';
import {MulticlassPreds} from '../lib/lit_types';
import {GroupedExamples, ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {doesOutputSpecContain, getMarginFromThreshold, getThresholdFromMargin, findSpecKeys, isBinaryClassification} from '../lib/utils';
import {ClassificationService, GroupService} from '../services/services';
import {NumericFeatureBins} from '../services/group_service';
import {styles as sharedStyles} from '../lib/shared_styles.css';


/** Margins for each optimization type. */
interface CaculatedMargins {
  [optimizer: string]: number;
}

/** Any facet of a dataset can have its own margin value. */
interface CalculatedMarginsPerFacet {
  [facetString: string]: CaculatedMargins;
}

/** Each output field has its own margin settings. */
interface CalculatedMarginsPerField {
  [fieldName: string]: CalculatedMarginsPerFacet;
}

/**
 * A LIT module that renders regression results.
 */
@customElement('thresholder-module')
export class ThresholderModule extends LitModule {
  static override title = 'Binary Classifier Thresholds';
  static override referenceURL =
      'https://github.com/PAIR-code/lit/wiki/components.md#binary-classification-thresholds';
  static override numCols = 3;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`
  <thresholder-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </thresholder-module>`;

  static override get styles() {
    return [
      sharedStyles, css`
        .cost-ratio-input {
          width: 50px;
        }

        .module-toolbar {
          margin-bottom: 4px;
        }
    `
    ];
  }

  // Cost ratio of false positives to false negatives to use in calculating
  // optimal thresholds.
  private costRatio = 1;

  private selectedFacetBins: NumericFeatureBins = {};

  // Selected features to create faceted thresholds from.
  @observable private readonly selectedFacets: string[] = [];

  // Hold all calculated margins from the tresholder component.
  @observable private calculatedMargins: CalculatedMarginsPerField = {};

  private readonly classificationService =
    app.getService(ClassificationService);
  private readonly groupService = app.getService(GroupService);
  private readonly facetingControl = document.createElement('faceting-control');

  constructor() {
    super();

    const facetsChange = (event: CustomEvent<FacetsChange>) => {
      this.selectedFacets.length = 0;
      this.selectedFacets.push(...event.detail.features);
      this.selectedFacetBins = event.detail.bins;
    };
    this.facetingControl.contextName = ThresholderModule.title;
    this.facetingControl.addEventListener(
        'facets-change', facetsChange as EventListener);
  }

  override firstUpdated() {
    const getGroupedExamples = () => this.groupedExamples;
    this.reactImmediately(
        getGroupedExamples, groupedExamples => {
          this.updateMarginCategories(groupedExamples);
        });
  }

  /**
   * Set the facets for which margins can be individually set based on the
   * facet groups selected.
   */
  private updateMarginCategories(groupedExamples: GroupedExamples) {
    this.calculatedMargins = {};
    for (const predKey of this.binaryClassificationKeys) {
      this.classificationService.setMarginGroups(
          this.model, predKey, groupedExamples);
      const marginsForKey: CalculatedMarginsPerFacet = {};
      for (const facets of Object.keys(groupedExamples)) {
        marginsForKey[facets] = {};
      }
      this.calculatedMargins[predKey] = marginsForKey;
    }
  }

  @computed
  private get binaryClassificationKeys() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const classificationKeys = findSpecKeys(outputSpec, MulticlassPreds);
    return classificationKeys.filter(
        key => isBinaryClassification(outputSpec[key]));
  }

  private async calculateThresholds() {
    const config = {
      'cost_ratio': this.costRatio,
      'facets': this.groupedExamples
    };
    const thresholds = await this.apiService.getInterpretations(
        this.appState.currentInputData, this.model,
        this.appState.currentDataset, 'thresholder', config);
    // Add returned faceted thresholds to the store calculated margins.
    for (const thresholdResults of thresholds) {
      const marginsForKey =
          this.calculatedMargins[thresholdResults['pred_key']];
      for (const facetedThresholds of Object.keys(
               thresholdResults['thresholds'])) {
        const marginsForThresholdTypes: CaculatedMargins = {};
        for (const thresholdType of Object.keys(
                 thresholdResults['thresholds'][facetedThresholds])) {
          marginsForThresholdTypes[thresholdType] = getMarginFromThreshold(
              thresholdResults['thresholds'][facetedThresholds][thresholdType]);
        }
        marginsForKey[facetedThresholds] = marginsForThresholdTypes;
      }
      this.calculatedMargins[thresholdResults['pred_key']] = marginsForKey;
    }
  }

  /** The facet groups created by the feature selector checkboxes. */
  @computed
  private get groupedExamples() {
    // Get the intersectional feature bins.
    const groupedExamples = this.groupService.groupExamplesByFeatures(
        this.selectedFacetBins,
        this.appState.currentInputData,
        this.selectedFacets);
    return groupedExamples;
  }

  renderTable(predKey: string) {
    const buttonStyles = {
      'background': 'transparent',
      'border-radius': '4px',
      'border': '1px solid rgb(189, 193, 198)',
      'color': 'rgb(26, 115, 232)'
    };
    const columnNames: Array<string|ColumnHeader> = ['Facet'];
    const margins = this.calculatedMargins[predKey];
    if (margins == null) {
      return null;
    }
    let firstFacet = true;
    const tableRows: TableEntry[][] = [];
    for (const facetKey of Object.keys(margins)) {
      const facetKeyDisplay = facetKey === '' ? 'All' : facetKey;
      const row: TableEntry[] = [facetKeyDisplay];
      for (const thresholdType of Object.keys(margins[facetKey])) {
        if (firstFacet) {
          // For the table header, add names of different optimized thresholds
          // and buttons to apply them.
          const applyThreshold = () => {
            for (const fKey of Object.keys(margins)) {
              this.classificationService.setMargin(
                this.model, predKey, margins[fKey][thresholdType],
                this.groupedExamples[fKey]);
            }
          };
          columnNames.push({
            name: thresholdType,
            html: html`
              <div>${thresholdType}</div>
              <button style=${styleMap(buttonStyles)} @click=${applyThreshold}>
                Apply
              </button>`,
            rightAlign: true
          });
        }
        row.push(getThresholdFromMargin(
            margins[facetKey][thresholdType]).toFixed(2));
      }
      const margin = this.classificationService.getMargin(
          this.model, predKey, this.groupedExamples[facetKey]);
      const callback = (e: CustomEvent<ThresholdChange>) => {
        this.classificationService.setMargin(
            this.model, predKey, e.detail.margin,
            this.groupedExamples[facetKey]);
      };
      row.push(html`<threshold-slider .margin=${margin} label=${facetKey}
                    ?isThreshold=${true} @threshold-changed=${callback}>
                  </threshold-slider>`);
      firstFacet = false;
      tableRows.push(row);
    }
    columnNames.push(predKey + ' threshold');
    return html`
        <lit-data-table
          .columnNames=${columnNames}
          .data=${tableRows}
        ></lit-data-table>
    `;
  }

  renderControls() {
    const handleCostRatioInput = (e: InputEvent) => {
      this.costRatio = +((e.target as HTMLInputElement).value);
    };

    const costRatioTooltip = "The cost of false positives relative to false " +
        "negatives. Used to find optimal classifier thresholds";
    const calculateTooltip = "Calculate optimal threholds for each facet " +
        "using the cost ratio and a number of different techniques";
    return html`
        ${this.facetingControl}
        <div title=${costRatioTooltip}>Cost ratio (FP/FN):</div>
        <input type=number class="cost-ratio-input" step="0.1" min=0 max=20
               .value=${this.costRatio.toString()}
               @input=${handleCostRatioInput}>
        <button class='hairline-button' title=${calculateTooltip}
           @click=${this.calculateThresholds}>
          Get optimal thresholds
        </button>`;
  }

  override renderImpl() {
    const tables =
        this.binaryClassificationKeys.map(key => this.renderTable(key));

    return html`
        <div class='module-container'>
          <div class='module-toolbar'>
            ${this.renderControls()}
          </div>
          <div class='module-results-area ${SCROLL_SYNC_CSS_CLASS}'>
            ${tables}
          </div>
        </div>
        `;
  }

  static override shouldDisplayModule(
      modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(
        modelSpecs, MulticlassPreds, isBinaryClassification);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'thresholder-module': ThresholderModule;
  }
}
