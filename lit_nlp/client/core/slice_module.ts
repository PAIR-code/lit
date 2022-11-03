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
import {html} from 'lit';
import {classMap} from 'lit/directives/class-map';
import {computed, observable} from 'mobx';

import {app} from './app';
import {LitModule} from './lit_module';
import {ModelInfoMap, Spec} from '../lib/types';
import {handleEnterKey} from '../lib/utils';
import {GroupService, NumericFeatureBins} from '../services/group_service';
import {SliceService} from '../services/services';
import {STARRED_SLICE_NAME} from '../services/slice_service';
import {FacetsChange} from '../core/faceting_control';


import {styles as sharedStyles} from '../lib/shared_styles.css';
import {styles} from './slice_module.css';

/**
 * The slice controls module
 */
@customElement('lit-slice-module')
export class SliceModule extends LitModule {
  static override get styles() {
    return [sharedStyles, styles];
  }

  static override title = 'Slice Editor';
  static override referenceURL =
      'https://github.com/PAIR-code/lit/wiki/ui_guide.md#slices';
  static override numCols = 2;
  static override collapseByDefault = true;
  static override duplicateForModelComparison = false;

  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`
  <lit-slice-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </lit-slice-module>`;

  private readonly sliceService = app.getService(SliceService);
  private readonly groupService = app.getService(GroupService);
  private readonly facetingControl = document.createElement('faceting-control');
  private sliceByBins: NumericFeatureBins = {};

  @observable private sliceByFeatures: string[] = [];

  @observable private sliceName: string|null = null;

  constructor() {
    super();

    const facetsChange = (event: CustomEvent<FacetsChange>) => {
      this.sliceByFeatures = event.detail.features;
      this.sliceByBins = event.detail.bins;
    };
    this.facetingControl.contextName = SliceModule.title;
    this.facetingControl.addEventListener(
        'facets-change', facetsChange as EventListener);
  }

  @computed
  private get createButtonEnabled() {
    const sliceFromFilters =
        (this.selectionService.selectedIds.length === 0 &&
         this.anyCheckboxChecked);
    return (
        // Making a slice from filters (name generated based on filters).
        sliceFromFilters ||
        // Making a slice from selected points (must give a name)
        (this.sliceName !== null) && (this.sliceName !== '') &&
            (this.selectionService.selectedIds.length > 0));
  }

  @computed
  private get anyCheckboxChecked() {
    return this.sliceByFeatures.length > 0;
  }


  private lastCreatedSlice() {
    const allSlices = this.sliceService.sliceNames;
    return allSlices[allSlices.length - 1];
  }

  private handleClickCreate() {
    if (this.anyCheckboxChecked) {
      this.makeSlicesFromAllLabelCombos();
      this.selectSlice(this.lastCreatedSlice());
    } else {
      const selectedIds = this.selectionService.selectedIds;
      const createSliceName = this.sliceName;
      if (createSliceName != null) {
        this.sliceService.addNamedSlice(createSliceName, selectedIds);
      }
      this.selectSlice(createSliceName);
    }
    this.sliceName = null;
    this.sliceByFeatures = [];
  }

  /**
   * Make slices from all combinations of the selected features.
   */
  private makeSlicesFromAllLabelCombos() {
    const data = this.selectionService.selectedOrAllInputData;
    const namedSlices = this.groupService.groupExamplesByFeatures(
                        this.sliceByBins, data, this.sliceByFeatures);

    // Make a slice per combination.
    const sliceNamePrefix = (this.sliceName == null) ? '' : this.sliceName + ' ';
    Object.keys(namedSlices).forEach(sliceName => {
      const createSlicename = `${sliceNamePrefix}${sliceName}`;
      const ids = namedSlices[sliceName].data.map(d => d.id);
      this.sliceService.addNamedSlice(createSlicename, ids);
    });
  }

  private selectSlice(sliceName: string|null) {
    this.sliceService.selectNamedSlice(sliceName, this);
  }

  renderCreate() {
    const onClickCreate = () => {
      this.handleClickCreate();
    };
    const onInputChange = (e: Event) => {
      // tslint:disable-next-line:no-any
      this.sliceName = (e as any).target.value;
    };

    const onKeyUp = (e: KeyboardEvent) => {
      handleEnterKey(e, onClickCreate);
    };
    // clang-format off
    return html`
      <div class="row-container">
        <input type="text" id="input-box" .value=${this.sliceName}
          placeholder="Enter name" @input=${onInputChange}
          @keyup=${(e: KeyboardEvent) => {onKeyUp(e);}}/>
        <button class='hairline-button'
          ?disabled="${!this.createButtonEnabled}"
          @click=${onClickCreate}>${this.sliceByFeatures.length > 0 ?
          'Create slices': 'Create slice'}
        </button>
      </div>
    `;
    // clang-format on
  }

  renderSliceRow(sliceName: string) {
    const selectedSliceName = this.sliceService.selectedSliceName;
    const itemClass = classMap(
        {'selector-item': true, 'selected': sliceName === selectedSliceName});
    const itemClicked = () => {
      const newSliceName = selectedSliceName === sliceName ? null : sliceName;
      this.selectSlice(newSliceName);
    };
    const numDatapoints =
        this.sliceService.getSliceByName(sliceName)?.length ?? 0;

    // Only enable appending if there are new examples to add.
    const appendButtonEnabled =
        this.selectionService.selectedIds
            .filter(id => !this.sliceService.isInSlice(sliceName, id))
            .length > 0;
    const appendIconClass =
        classMap({'icon-button': true, 'disabled': !appendButtonEnabled});
    const appendClicked = (e: Event) => {
      e.stopPropagation(); /* don't select row */
      this.sliceService.addIdsToSlice(
          sliceName, this.selectionService.selectedIds);
    };


    const deleteClicked = (e: Event) => {
      e.stopPropagation(); /* don't select row */
      this.sliceService.deleteNamedSlice(sliceName);
    };

    const clearIconClass = classMap({
      'icon-button': true,
      'mdi-outlined': true,
      'disabled': numDatapoints <= 0
    });
    const clearClicked = (e: Event) => {
      e.stopPropagation(); /* don't select row */
      const ids = this.sliceService.getSliceByName(sliceName) ?? [];
      this.sliceService.removeIdsFromSlice(sliceName, ids);
    };

    // clang-format off
    return html`
      <div class=${itemClass} @click=${itemClicked}>
        <span class='slice-name'>${sliceName}</span>
        <span class="number-label">
          ${numDatapoints} ${numDatapoints === 1 ? 'datapoint' : 'datapoints'}
          <mwc-icon class=${appendIconClass} @click=${appendClicked}
           title="Add selected to this slice">
           add_circle_outline
          </mwc-icon>
          ${sliceName === STARRED_SLICE_NAME ?
            html`<mwc-icon class=${clearIconClass} @click=${clearClicked}
                  title="Reset this slice">
                   clear
                 </mwc-icon>` :
            html`<mwc-icon class='icon-button' @click=${deleteClicked}
                  title="Delete this slice">
                   delete_outline
                 </mwc-icon>`}
        </span>
      </div>`;
    // clang-format on
  }

  renderSliceSelector() {
    // clang-format off
    return html`
      <div id="select-container">
        <label>Select slice</label>
        <div id="slice-selector">
          ${this.sliceService.sliceNames.map(sliceName =>
            this.renderSliceRow(sliceName)
          )}
        </div>
      </div>
    `;
    // clang-format on
  }

  override renderImpl() {
    // clang-format off
    return html`
      <div class='module-container'>
        ${this.renderCreate()}
        <div class="row-container" >
          ${this.facetingControl}
        </div>
        ${this.renderSliceSelector()}
      </div>
    `;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-slice-module': SliceModule;
  }
}
