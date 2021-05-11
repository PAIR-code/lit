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
import {customElement, html} from 'lit-element';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {MatrixCell} from '../elements/data_matrix';
import {IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, objToDictKey} from '../lib/utils';
import {ClassificationInfo} from '../services/classification_service';
import {GetFeatureFunc, GroupService} from '../services/group_service';
import {ClassificationService} from '../services/services';

import {styles} from './confusion_matrix_module.css';
import {styles as sharedStyles} from './shared_styles.css';


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
  static title = 'Confusion Matrix';
  static template = (model = '') => {
    return html`
      <confusion-matrix-module model=${model}>
      </confusion-matrix-module>`;
  };
  static numCols = 3;
  static duplicateForModelComparison = false;

  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly groupService = app.getService(GroupService);

  @observable verticalColumnLabels = false;
  @observable hideEmptyLabels = false;
  // These are not observable, because we don't want to trigger a re-render
  // until the matrix cells are updated asynchronously.
  selectedRowOption = 0;
  selectedColOption = 0;

  // Output state for rendering. Computed asynchronously.
  @observable matrixCells: MatrixCell[][] = [];

  constructor() {
    super();
    this.setInitialOptions();
  }

  firstUpdated() {
    // Calculate the initial confusion matrix.
    const getCurrentInputData = () => this.appState.currentInputData;
    this.react(getCurrentInputData, currentInputData => {
      this.calculateMatrix();
    });

    const getMarginSettings = () =>
        this.classificationService.allMarginSettings;
    this.react(getMarginSettings, margins => {
      this.calculateMatrix();
    });

    const getSelectedInputData = () => this.selectionService.selectedInputData;
    this.react(getSelectedInputData, selectedInputData => {
      // Don't reset if we just clicked a cell from this module.
      if (this.selectionService.lastUser !== this) {
        this.calculateMatrix();
      }
    });

    // Update once on init, to avoid duplicate calls.
    this.calculateMatrix();
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
      const getLabelsFn = (d: IndexedInput, i: number) =>
          this.groupService.getFeatureValForInput(d, i, labelKey);
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

  /**
   * Set the matrix cell information based on the selected axes and examples.
   */
  private async calculateMatrix() {
    const rowOption = this.options[this.selectedRowOption];
    const colOption = this.options[this.selectedColOption];

    const rowLabels = rowOption.labelList;
    const colLabels = colOption.labelList;

    const rowName = rowOption.name;
    const colName = colOption.name;

    const data = this.selectionService.selectedOrAllInputData;

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
    const getFeatFunc: GetFeatureFunc = (d, i, key) => resultsDict[key][i];
    const bins = this.groupService.groupExamplesByFeatures(
        data, [rowName, colName], getFeatFunc);

    this.matrixCells = rowLabels.map(rowLabel => {
      return colLabels.map(colLabel => {
        // If the rows and columns are the same feature but the cells are for
        // different values of that feature, then by definition no examples can
        // go into that cell. Handle this special case as the facetsDict below
        // only handles a single value per feature.
        if (colName === rowName && colLabel !== rowLabel) {
          return {ids: [], selected: false};
        }
        // Find the bin corresponding to this row/column value combination.
        const facetsDict = {[colName]: colLabel, [rowName]: rowLabel};
        const bin = bins[objToDictKey(facetsDict)];
        const ids = bin ? bin.data.map(example => example.id) : [];
        return {ids, selected: false};
      });
    });
  }

  render() {
    // clang-format off
    return html`
      <div class='module-container'>
        <div class='module-toolbar multiline-toolbar'>
          ${this.renderControls()}
        </div>
        <div class='module-results-area'>
          ${this.renderMatrix()}
        </div>
      </div>
    `;
    // clang-format on
  }

  private renderControls() {
    const rowChange = (e: Event) => {
      this.selectedRowOption = +((e.target as HTMLSelectElement).value);
      this.calculateMatrix();
    };
    const colChange = (e: Event) => {
      this.selectedColOption = +((e.target as HTMLSelectElement).value);
      this.calculateMatrix();
    };
    const toggleHideCheckbox = () => {
      this.hideEmptyLabels = !this.hideEmptyLabels;
      this.calculateMatrix();
    };
    return html`
      <div class="dropdown-holder">
        <label class="dropdown-label">Rows</label>
        <select class="dropdown" @change=${rowChange}>
          ${this.options.map((option, i) => html`
            <option ?selected=${this.selectedRowOption === i} value=${i}>
              ${option.name}
            </option>`)}
        </select>
      </div>
      <div class="dropdown-holder">
        <label class="dropdown-label">Columns</label>
        <select class="dropdown" @change=${colChange}>
          ${this.options.map((option, i) => html`
            <option ?selected=${this.selectedColOption === i} value=${i}>
              ${option.name}
             </option>`)}
        </select>
      </div>
      <lit-checkbox
        label="Hide empty labels"
        ?checked=${this.hideEmptyLabels}
        @change=${toggleHideCheckbox}>
      </lit-checkbox>
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

  private renderMatrix() {
    const rowOption = this.options[this.selectedRowOption];
    const colOption = this.options[this.selectedColOption];
    const rowLabels = rowOption.labelList;
    const colLabels = colOption.labelList;
    const rowTitle = rowOption.name;
    const colTitle = colOption.name;

    // Add event listener for selection events.
    const onCellClick = (event: Event) => {
      // tslint:disable-next-line:no-any
      const ids: string[] = (event as any).detail.ids;
      this.selectionService.selectIds(ids, this);
    };

    // clang-format off
    return html`
        <data-matrix
          .matrixCells=${this.matrixCells}
          ?hideEmptyLabels=${this.hideEmptyLabels}
          .rowTitle=${rowTitle}
          .rowLabels=${rowLabels}
          .colTitle=${colTitle}
          .colLabels=${colLabels}
          @matrix-selection=${onCellClick}
        >
        </data-matrix>
        `;
    // clang-format on
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'MulticlassPreds');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'confusion-matrix-module': ConfusionMatrixModule;
  }
}
