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
import {customElement, property} from 'lit/decorators';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {FacetsChange} from '../core/faceting_control';
import {LitModule} from '../core/lit_module';
import {MatrixCell, MatrixSelection} from '../elements/data_matrix';
import {ReactiveElement} from '../lib/elements';
import {MulticlassPreds} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {GroupedExamples, IndexedInput, ModelInfoMap} from '../lib/types';
import {arrayContainsSame, doesOutputSpecContain, facetMapToDictKey} from '../lib/utils';
import {FacetingMethod, GetFeatureFunc, GroupService, NumericFeatureBins} from '../services/group_service';
import {DataService, SelectionService} from '../services/services';

import {styles} from './confusion_matrix_module.css';


/**
 * Option for extracting labels to compare.
 */
interface CmatOption {
  name: string;
  labelList: string[];
  // Runner maps dataset to labels. Can involve backend calls.
  runner: (d: IndexedInput[]) => Promise<Array<(number | string | null)>>;
  parent?: string;  // parent field, used to set default selection
}

/**
 * A LIT module that renders confusion matrices for classification models.
 */
@customElement('confusion-matrix-module')
export class ConfusionMatrixModule extends LitModule {
  static override title = 'Confusion Matrix';
  static override referenceURL =
      'https://github.com/PAIR-code/lit/wiki/components.md#confusion-matrix';
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => {
        return html`
  <confusion-matrix-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </confusion-matrix-module>`;
      };
  static override numCols = 4;
  static override duplicateForModelComparison = false;

  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly dataService = app.getService(DataService);
  private readonly groupService = app.getService(GroupService);
  private readonly facetingControl = document.createElement('faceting-control');

  @observable verticalColumnLabels = false;
  @observable hideEmptyLabels = false;
  @observable showSelection = false;
  @observable selectedRowOption = 0;
  @observable selectedColOption = 1;

  @observable private facetFeatures: string[] = [];
  @observable private facetBins: NumericFeatureBins = {};

  // Maximum allowed entries in a row or column feature that can be selected.
  // TODO(lit-dev): Fix b/199503959 to remove this limitation.
  private readonly MAX_ENTRIES = 50;

  // Map of matrices for rendering. Computed asynchronously.
  @observable matrices: {[id: string]: MatrixCell[][]} = {};

  constructor() {
    super();
    this.setInitialOptions();

    const facetsChange = (event: CustomEvent<FacetsChange>) => {
      this.facetFeatures = event.detail.features;
      this.facetBins = event.detail.bins;
    };
    this.facetingControl.contextName = ConfusionMatrixModule.title;
    this.facetingControl.addEventListener(
        'facets-change', facetsChange as EventListener);
  }

  private setInitialOptions() {
    // Set default selection to be rows = labels, cols = preds of first model.
    for (let i = 0; i < this.options.length; i++) {
      // Use the 'parent' field to check if this option represents predictions.
      const parentName = this.options[i].parent;
      if (parentName !== undefined) {
        // Find the index of the option for this label, if present.
        for (let j = 0; j < this.options.length; j++) {
          if (this.options[j].name === parentName) {
            // Set row selection to the corresponding label field.
            this.selectedRowOption = j;
            break;
          }
        }
        // Set column selection to first 'model' selector.
        this.selectedColOption = i;
        break;
      }
    }
  }

  @computed
  private get shouldDisplaySelectionMatrix(): boolean {
    return this.showSelection &&
           this.selectionService.selectedInputData.length > 0;
  }

  /**
   * Data subsets, starting with the entire dataset, derived from the faceted
   * features. Each subset is rendered as its own Confusion Matrix.
   */
  @computed
  private get subsets(): Map<string, IndexedInput[]> {
    // Map structures iterate their .entries() in insertion order, so we
    // construct them in a certain order here
    const groups = new Map<string, IndexedInput[]>();

    // Always display the entire dataset and insert it first.
    const baseData = this.appState.currentInputData;
    groups.set(`Dataset (${baseData.length})`, baseData);

    // If there is a selection to show, insert it second
    if (this.shouldDisplaySelectionMatrix) {
      const selected = this.selectionService.selectedInputData;
      groups.set(`Selection (${selected.length})`, selected);
    }

    // Finally, insert the faceted subsets in alphabetical order of their keys
    if (this.facetFeatures.length > 0) {
      const groupedExamples =
          this.groupService.groupExamplesByFeatures(this.facetBins, baseData,
                                                    this.facetFeatures);
      const facetNames = Object.keys(groupedExamples).sort();

      for (const facet of facetNames) {
        const {data} = groupedExamples[facet];
        const label = `Dataset faceted by ${facet} (${data.length})`;
        groups.set(label, data);
      }
    }

    return groups;
  }

  /**
   * Options for the axes of the confusion matrix, such as ground truth label
   * or a model's predictions.
   */
  @computed
  get options(): CmatOption[] {
    // Get all preds fields and their parents
    const options: CmatOption[] = [];
    // TODO(b/156100081): Get proper reactions on data service columns.
    // tslint:disable:no-unused-variable Causes recompute on change.
    const data = this.dataService.dataVals;

    // From the data, we can bin by any categorical feature.
    const categoricalFeatures = this.groupService.categoricalFeatures;
    for (const labelKey of Object.keys(categoricalFeatures)) {
      const labelList = categoricalFeatures[labelKey];
      if (labelList.length > this.MAX_ENTRIES) {
        continue;
      }
      const bins = this.groupService.numericalFeatureBins([{
        featureName: labelKey,
        method: FacetingMethod.DISCRETE
      }]);
      const getLabelsFn = (d: IndexedInput, i: number) =>
          this.groupService.getFeatureValForInput(bins, d, labelKey);
      const labelsRunner = async (dataset: IndexedInput[]) =>
          dataset.map(getLabelsFn);
      options.push({name: labelKey, labelList, runner: labelsRunner});
    }

    return options;
  }

  override renderImpl() {
    const row = this.options[this.selectedRowOption];
    const col = this.options[this.selectedColOption];

    // clang-format off
    const matrices = [...this.subsets.entries()].map(([name, data]) =>
        html` <confusion-matrix ?hideEmptyLabels=${this.hideEmptyLabels}
                                .row=${row} .col=${col}
                                .data=${data} .label=${name}>
              </confusion-matrix>`);

    return html`<div class="module-container">
                  <div class="module-toolbar">${this.renderControls()}</div>
                  <div class="module-results-area">
                    <div class="non-faceted-matrix">
                      ${matrices.shift()}
                      ${this.shouldDisplaySelectionMatrix ? matrices.shift() :
                                                            null}
                    </div>
                    <div class="faceted-matrices">${matrices}</div>
                  </div>
                </div>`;
    // clang-format on
  }

  private renderControls() {
    const rowChange = (e: Event) => {
      this.selectedRowOption = Number((e.target as HTMLSelectElement).value);
    };
    const colChange = (e: Event) => {
      this.selectedColOption = Number((e.target as HTMLSelectElement).value);
    };
    const toggleSelectionCheckbox = () => {
      this.showSelection = !this.showSelection;
    };
    const toggleHideCheckbox = () => {
      this.hideEmptyLabels = !this.hideEmptyLabels;
    };
    // clang-format off
    return html`<div class="matrix-options">
                  <lit-checkbox label="Show matrix for selection"
                                ?checked=${this.showSelection}
                                @change=${toggleSelectionCheckbox}>
                  </lit-checkbox>
                  <lit-checkbox label="Hide empty labels"
                                ?checked=${this.hideEmptyLabels}
                                @change=${toggleHideCheckbox}>
                  </lit-checkbox>
                  <div class="spacer"></div>
                  <div class="dropdown-holder">
                    <label class="dropdown-label">Rows:</label>
                    <select class="dropdown" @change=${rowChange}>
                      ${this.options.map((option, i) => html`
                        <option value=${i}
                                ?selected=${this.selectedRowOption === i}
                                ?disabled=${i === this.selectedColOption}>
                          ${option.name}
                        </option>`)}
                    </select>
                  </div>
                  <div class="dropdown-holder">
                    <label class="dropdown-label">Columns:</label>
                    <select class="dropdown" @change=${colChange}>
                      ${this.options.map((option, i) => html`
                        <option value=${i}
                                ?selected=${this.selectedColOption === i}
                                ?disabled=${i === this.selectedRowOption}>
                          ${option.name}
                          </option>`)}
                    </select>
                  </div>
                  <div class="spacer"></div>
                  ${this.facetingControl}
                </div>`;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap) {
    return doesOutputSpecContain(modelSpecs, MulticlassPreds);
  }
}

/**
 * Computes and renders a confusion matrix given a row feature, column feature,
 * and subset of data.
 */
@customElement('confusion-matrix')
class ConfusionMatrix extends ReactiveElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  @observable @property({type: Boolean}) hideEmptyLabels = false;
  /** Feature to use for the rows of the matrix */
  @observable @property({type: Object}) row?: CmatOption;
  /** Feature to use for the columns of the matrix */
  @observable @property({type: Object}) col?: CmatOption;
  /** Dataset to map into the cells of the matrix */
  @observable @property({type: Array}) data?: IndexedInput[];
  /** Label for the matrix */
  @observable @property({type: String}) label = 'Confusion Matrix';

  @observable private cells: MatrixCell[][] = [];


  // tslint:disable-next-line:no-any
  private readonly latestLoadPromises = new Map<string, Promise<any>>();
  private groups: GroupedExamples = {};
  private readonly groupService = app.getService(GroupService);
  private readonly selectionService = app.getService(SelectionService);

  private async loadLatest<T>(key: string, promise: Promise<T>) {
    this.latestLoadPromises.set(key, promise);

    const result = await promise;

    if (this.latestLoadPromises.get(key) === promise) {
      this.latestLoadPromises.delete(key);
      return result;
    }

    return null;
  }

  private async calculateMatrix() {
    // If there is no data loaded, then do not attempt to create a matrix.
    if (this.row == null || this.col == null || this.data == null ||
        this.data.length === 0) return;

    // Runners may make backend calls, so wrap them in promises.
    const promise = Promise.all([
      this.row.runner(this.data),
      this.col.runner(this.data)
    ]);
    const results = await this.loadLatest('rowColIdxs', promise);
    if (results == null) return;

    // If label lists for rows/cols are identical, ensure they are rendered in
    // the same order.
    const labelsIdentical = arrayContainsSame(
        this.row.labelList, this.col.labelList);
    if (labelsIdentical) {
      this.col.labelList = this.row.labelList;
    }

    const rowLabels = this.row.labelList;
    const colLabels = this.col.labelList;
    const rowName = this.row.name;
    const colName = this.col.name;
    const resultsDict = {[rowName]: results[0], [colName]: results[1]};

    // Since groupService.groupExamplesByFeatures only uses indexedInput data
    // properties by default, provide a custom function that also supports the
    // calculated predicted properties.
    const getFeatFunc: GetFeatureFunc = (b, d, i, feat) => resultsDict[feat][i];
    const bins = this.groupService.numericalFeatureBins([
      {featureName: rowName, method: FacetingMethod.DISCRETE},
      {featureName: colName, method: FacetingMethod.DISCRETE}
    ]);
    this.groups = this.groupService.groupExamplesByFeatures(
        bins, this.data, [rowName, colName], getFeatFunc);

    this.cells = rowLabels.map(rowLabel => colLabels.map(colLabel => {
      // If the rows and columns are the same feature but the cells are for
      // different values of that feature, then by definition no examples can
      // go into that cell. Handle this special case as the facetsDict below
      // only handles a single value per feature.
      if (colName === rowName && colLabel !== rowLabel) {
        return {size: 0, selected: false};
      }
      // Find the bin corresponding to this row/column value combination.
      const facetsDict = {
        [colName]: {val: colLabel, displayVal: colLabel},
        [rowName]: {val: rowLabel, displayVal: rowLabel}
      };
      const bin = this.groups[facetMapToDictKey(facetsDict)];
      return {size: bin ? bin.data.length : 0, selected: false} as MatrixCell;
    }));
  }

  override firstUpdated() {
    const configChange = () => [this.row, this.col, this.data];
    this.reactImmediately(configChange, () => {this.calculateMatrix();});
  }

  override render() {
    if (this.row == null || this.col == null || this.data == null ||
        this.data.length === 0) {
      return html``;
    }

    const onCellClick = (event: CustomEvent<MatrixSelection>) => {
      event.stopPropagation();
      event.preventDefault();

      let ids: string[] = [];
      for (const [i, j] of event.detail.cells) {
        const rowLabel = this.row!.labelList[i];
        const colLabel = this.col!.labelList[j];
        const facetsDict = {
          [this.col!.name]: {val: colLabel, displayVal: colLabel},
          [this.row!.name]: {val: rowLabel, displayVal: rowLabel}
        };
        const bin = this.groups[facetMapToDictKey(facetsDict)];
        ids = ids.concat(bin.data.map(example => example.id));
      }
      this.selectionService.selectIds(ids, this);
    };

    return html`
        <div class='confusion-matrix'>
          <div class="matrix-label">${this.label}</div>
          <data-matrix class='matrix'
                      ?hideEmptyLabels=${this.hideEmptyLabels}
                      .matrixCells=${this.cells}
                      .rowTitle=${this.row.name}
                      .rowLabels=${this.row.labelList}
                      .colTitle=${this.col.name}
                      .colLabels=${this.col.labelList}
                      @matrix-selection=${onCellClick}></data-matrix>
        </div>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'confusion-matrix': ConfusionMatrix;
    'confusion-matrix-module': ConfusionMatrixModule;
  }
}
