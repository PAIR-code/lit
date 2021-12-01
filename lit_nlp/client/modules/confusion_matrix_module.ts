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
import {MatrixCell} from '../elements/data_matrix';
import {IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, facetMapToDictKey, findSpecKeys} from '../lib/utils';
import {ClassificationInfo} from '../services/classification_service';
import {GetFeatureFunc, GroupService, FacetingMethod} from '../services/group_service';
import {ClassificationService} from '../services/services';

import {styles} from './confusion_matrix_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';


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
  static override template = (model = '') => {
    return html`
      <confusion-matrix-module model=${model}>
      </confusion-matrix-module>`;
  };
  static override numCols = 3;
  static override duplicateForModelComparison = false;

  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly groupService = app.getService(GroupService);

  @observable verticalColumnLabels = false;
  @observable hideEmptyLabels = false;
  @observable updateOnSelection = true;
  @observable selectedRowOption = 0;
  @observable selectedColOption = 0;

  private lastSelectedRow = -1;
  private lastSelectedCol = -1;

  // Maximum allowed entries in a row or column feature that can be selected.
  // TODO(lit-dev): Fix b/199503959 to remove this limitation.
  private readonly MAX_ENTRIES = 50;

  // Map of matrices for rendering. Computed asynchronously.
  @observable matrices: {[id: string]: MatrixCell[][]} = {};

  constructor() {
    super();
    this.setInitialOptions();
  }

  override firstUpdated() {
    // Calculate the initial confusion matrix.
    const getCurrentInputData = () => this.appState.currentInputData;
    this.react(getCurrentInputData, currentInputData => {
      this.matrices = {};
      this.calculateMatrix(this.selectedRowOption, this.selectedColOption);
    });

    const getMarginSettings = () =>
        this.classificationService.allMarginSettings;
    this.react(getMarginSettings, margins => {
      this.updateMatrices();
    });
    const getUpdateOnSelection = () => this.updateOnSelection;
    this.react(getUpdateOnSelection, updateOnSelection => {
      this.updateMatrices();
    });

    const getSelectedInputData = () => this.selectionService.selectedInputData;
    this.react(getSelectedInputData, async selectedInputData => {
      // If the selection is from another module and this is set to update
      // on selection changes, then update the matrices.
      if (this.selectionService.lastUser !== this && this.updateOnSelection) {
        await this.updateMatrices();
      }
      // If the selection is from this module then update all matrices that
      // weren't the cause of the selection, to reset their selection states
      // and recalculate their cells if we are updating on selections.
      if (this.selectionService.lastUser === this) {
        for (const id of Object.keys(this.matrices)) {
          const [row, col] = this.getOptionsFromMatrixId(id);
          if (row !== this.lastSelectedRow || col !== this.lastSelectedCol) {
            await this.calculateMatrix(row, col);
          }
        }
      }
    });

    // Update once on init, to avoid duplicate calls.
    this.calculateMatrix(this.selectedRowOption, this.selectedColOption);
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

  /**
   * Options for the axes of the confusion matrix, such as ground truth label
   * or a model's predictions.
   */
  @computed
  get options(): CmatOption[] {
    // Get all preds fields and their parents
    const models = this.appState.currentModels;
    const options: CmatOption[] = [];

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

    // For each model, select preds fields.
    for (const model of models) {
      const outputSpec = this.appState.getModelSpec(model).output;
      for (const predKey of findSpecKeys(outputSpec, 'MulticlassPreds')) {
        // Labels for this key.
        const labelKey = outputSpec[predKey].parent;
        // Note: vocab should always be present for MulticlassPreds.
        const labelList = outputSpec[predKey].vocab!;
        if (labelList.length > this.MAX_ENTRIES) {
          continue;
        }
        // Preds for this key.
        const predsRunner = async (dataset: IndexedInput[]) => {
          const preds = await this.classificationService.getClassificationPreds(
              dataset, model, this.appState.currentDataset);
          const classPromises = preds.map(
              async (d: {[key: string]: number[]}, i: number) =>
                  this.classificationService.getResults(
                      [dataset[i].id], model, predKey));
          const classResults = await Promise.all(classPromises);
          return classResults.map(
              (info: ClassificationInfo[]) =>
                  labelList[info[0].predictedClassIdx]);
        };
        options.push({
          name: `${model}:${predKey}`,
          labelList,
          runner: predsRunner,
          parent: labelKey
        });
      }
    }

    return options;
  }

  private async updateMatrices() {
    for (const id of Object.keys(this.matrices)) {
      const [row, col] = this.getOptionsFromMatrixId(id);
      await this.calculateMatrix(row, col);
    }
  }

  /**
   * Set the matrix cell information based on the selected axes and examples.
   */
  private async calculateMatrix(row: number, col: number) {
    const rowOption = this.options[row];
    const colOption = this.options[col];

    const rowLabels = rowOption.labelList;
    const colLabels = colOption.labelList;

    const rowName = rowOption.name;
    const colName = colOption.name;

    // When updating on selection, use the selected data if a selection exists.
    // Otherwise use the whole dataset.
    const data = this.updateOnSelection ?
        this.selectionService.selectedOrAllInputData :
        this.appState.currentInputData;

    // If there is no data loaded, then do not attempt to create a confusion
    // matrix.
    if (data.length === 0) {
      return;
    }

    // These can make backend calls, or just extract labels from the input data.
    const rowIdxsPromise = rowOption.runner(data);
    const colIdxsPromise = colOption.runner(data);

    const promises = Promise.all([rowIdxsPromise, colIdxsPromise]);

    const results = await this.loadLatest('rowAndColIdxs', promises);
    if (results === null) return;

    const resultsDict = {[rowName]: results[0], [colName]: results[1]};

    // Since groupService.groupExamplesByFeatures only uses indexedInput data
    // properties by default, provide a custom function that also supports the
    // calculated predicted properties.
    const getFeatFunc: GetFeatureFunc = (b, d, i, feat) => resultsDict[feat][i];
    const bins = this.groupService.numericalFeatureBins([
      {featureName: rowName, method: FacetingMethod.DISCRETE},
      {featureName: colName, method: FacetingMethod.DISCRETE}
    ]);
    const groups = this.groupService.groupExamplesByFeatures(
        bins, data, [rowName, colName], getFeatFunc);

    const id = this.getMatrixId(row, col);
    const matrixCells = rowLabels.map(rowLabel => {
      return colLabels.map(colLabel => {
        // If the rows and columns are the same feature but the cells are for
        // different values of that feature, then by definition no examples can
        // go into that cell. Handle this special case as the facetsDict below
        // only handles a single value per feature.
        if (colName === rowName && colLabel !== rowLabel) {
          return {ids: [], selected: false};
        }
        // Find the bin corresponding to this row/column value combination.
        const facetsDict = {
          [colName]: {val:colLabel, displayVal: colLabel},
          [rowName]: {val:rowLabel, displayVal: rowLabel}};
        const bin = groups[facetMapToDictKey(facetsDict)];
        const ids = bin ? bin.data.map(example => example.id) : [];
        return {ids, selected: false};
      });
    });
    this.matrices[id] = matrixCells;
  }

  private getMatrixId(row: number, col: number) {
    return `${row}:${col}`;
  }

  private getOptionsFromMatrixId(id: string) {
    return id.split(":").map(numStr => +numStr);
  }

  private canCreateMatrix(row: number, col: number) {
    // Create create a matrix if the rows and columns are for different fields
    // and this matrix isn't already created.
    if (row === col) {
      return false;
    }
    const id = this.getMatrixId(row, col);
    return this.matrices[id] == null;
  }

  @computed
  get matrixCreateTooltip() {
    if (this.selectedRowOption === this.selectedColOption) {
      return 'Must set different row and column options';
    }
    if (this.matrices[
          this.getMatrixId(this.selectedRowOption, this.selectedColOption)] !=
        null) {
      return 'Matrix for current row and column options already exists';
    }
    return '';
  }

  override render() {
    const renderMatrices = () => {
      return Object.keys(this.matrices).map(id => {
        const [row, col] = this.getOptionsFromMatrixId(id);
        return this.renderMatrix(row, col, this.matrices[id]);
      });
    };
    // clang-format off
    return html`
      <div class='module-container'>
        <div class='module-toolbar'>
          ${this.renderControls()}
        </div>
        <div class='module-results-area matrices-holder'>
          ${renderMatrices()}
        </div>
      </div>
    `;
    // clang-format on
  }

  private renderControls() {
    const rowChange = (e: Event) => {
      this.selectedRowOption = +((e.target as HTMLSelectElement).value);
    };
    const colChange = (e: Event) => {
      this.selectedColOption = +((e.target as HTMLSelectElement).value);
    };
    const toggleUpdateOnSelection = () => {
      this.updateOnSelection = !this.updateOnSelection;
    };
    const toggleHideCheckbox = () => {
      this.hideEmptyLabels = !this.hideEmptyLabels;
    };
    const onCreateMatrix = () => {
      this.calculateMatrix(this.selectedRowOption, this.selectedColOption);
    };
    return html`
      <div class="flex">
        <lit-checkbox
          label="Update on selection"
          ?checked=${this.updateOnSelection}
          @change=${toggleUpdateOnSelection}>
        </lit-checkbox>
        <lit-checkbox
          label="Hide empty labels"
          ?checked=${this.hideEmptyLabels}
          @change=${toggleHideCheckbox}>
        </lit-checkbox>
      </div>
      <div class="flex">
        <div>
          <div class="dropdown-holder">
            <label class="dropdown-label">Rows:</label>
            <select class="dropdown" @change=${rowChange}>
              ${this.options.map((option, i) => html`
                <option ?selected=${this.selectedRowOption === i} value=${i}>
                  ${option.name}
                </option>`)}
            </select>
          </div>
          <div class="dropdown-holder">
            <label class="dropdown-label">Columns:</label>
            <select class="dropdown" @change=${colChange}>
              ${this.options.map((option, i) => html`
                <option ?selected=${this.selectedColOption === i} value=${i}>
                  ${option.name}
                 </option>`)}
            </select>
          </div>
        </div>
        <button class='hairline-button' id='create-button' @click=${onCreateMatrix}
          ?disabled="${!this.canCreateMatrix(this.selectedRowOption, this.selectedColOption)}"
          title=${this.matrixCreateTooltip}>
          Create matrix
        </button>
      </div>
    `;
  }

  renderColumnRotateButton() {
    const toggleVerticalColumnLabels = () => {
      this.verticalColumnLabels = !this.verticalColumnLabels;
    };

    // clang-format off
    return html`
      <mwc-icon-button-toggle class="icon-button"
        title="Rotate column labels"
        onIcon="text_rotate_up" offIcon="text_rotation_none"
        ?on="${this.verticalColumnLabels}"
        @MDCIconButtonToggle:change="${toggleVerticalColumnLabels}"
        @icon-button-toggle-change="${toggleVerticalColumnLabels}">
      </mwc-icon-button-toggle>
    `;
    // clang-format on
  }

  private renderMatrix(row: number, col: number, cells: MatrixCell[][]) {
    const rowOption = this.options[row];
    const colOption = this.options[col];
    const rowLabels = rowOption.labelList;
    const colLabels = colOption.labelList;
    const rowTitle = rowOption.name;
    const colTitle = colOption.name;

    // Add event listener for selection events.
    const onCellClick = (event: Event) => {
      // tslint:disable-next-line:no-any
      const ids: string[] = (event as any).detail.ids;
      this.lastSelectedRow = row;
      this.lastSelectedCol = col;
      this.selectionService.selectIds(ids, this);
    };
    // Add event listener for delete events.
    const onDelete = (event: Event) => {
      delete this.matrices[this.getMatrixId(row, col)];
    };

    // clang-format off
    return html`
        <data-matrix
          class='matrix'
          .matrixCells=${cells}
          ?hideEmptyLabels=${this.hideEmptyLabels}
          .rowTitle=${rowTitle}
          .rowLabels=${rowLabels}
          .colTitle=${colTitle}
          .colLabels=${colLabels}
          @matrix-selection=${onCellClick}
          @delete-matrix=${onDelete}
        >
        </data-matrix>
        `;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'MulticlassPreds');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'confusion-matrix-module': ConfusionMatrixModule;
  }
}
