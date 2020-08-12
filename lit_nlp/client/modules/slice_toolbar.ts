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
import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement, html} from 'lit-element';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {handleEnterKey} from '../lib/utils';
import {GroupService} from '../services/group_service';
import {AppState, SelectionService, SliceService} from '../services/services';

import {styles as sharedStyles} from './shared_styles.css';
import {styles} from './slice_toolbar.css';

/**
 * The slice controls toolbar
 */
@customElement('lit-slice-toolbar')
export class SliceToolbar extends MobxLitElement {
  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly selectionService = app.getService(SelectionService);
  private readonly sliceService = app.getService(SliceService);
  private readonly groupService = app.getService(GroupService);

  private readonly appState = app.getService(AppState);

  @observable private sliceByFeatures: string[] = [];

  @observable private sliceName: string = '';

  @computed
  private get createButtonEnabled() {
    const sliceFromFilters =
        (this.selectionService.selectedIds.length === 0 &&
         this.anyCheckboxChecked);
    return (
        // Making a slice from filters (name generated based on filters).
        sliceFromFilters ||
        // Making a slice from selected points (must give a name)
        (this.sliceName !== '') &&
            (this.selectionService.selectedIds.length > 0));
  }

  @computed
  private get deleteButtonEnabled() {
    return this.sliceService.selectedSliceName !== '';
  }

  @computed
  private get anyCheckboxChecked() {
    return this.sliceByFeatures.length;
  }


  @computed
  private get sliceNameInputEditable() {
    return (this.selectionService.selectedIds.length > 0) ||
        this.anyCheckboxChecked;
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
      this.sliceService.addNamedSlice(createSliceName, selectedIds);
      this.selectSlice(createSliceName);
    }
    this.sliceName = '';
    this.sliceByFeatures = [];
  }

  /**
   * Make slices from all combinations of the selected features.
   */
  private makeSlicesFromAllLabelCombos() {
    const data = this.selectionService.selectedOrAllInputData;
    const namedSlices =
        this.groupService.groupExamplesByFeatures(data, this.sliceByFeatures);

    // Make a slice per combination.
    const sliceNamePrefix = this.sliceName;
    Object.keys(namedSlices).forEach(sliceName => {
      const createSlicename = `${sliceNamePrefix}  ${sliceName}`;
      const ids = namedSlices[sliceName].data.map(d => d.id);
      this.sliceService.addNamedSlice(createSlicename, ids);
    });
  }

  private selectSlice(sliceName: string) {
    this.sliceService.selectNamedSlice(sliceName, this);
  }

  private deleteSlice() {
    this.sliceService.deleteNamedSlice(this.sliceService.selectedSliceName);
    this.selectSlice('');
  }

  renderButtons() {
    const onClickCreate = () => {
      this.handleClickCreate();
    };
    const onClickDelete = () => {
      this.deleteSlice();
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
      <input type="text" id="input-box" .value=${this.sliceName}
        placeholder="Enter name" @input=${onInputChange}
        ?readonly="${!this.sliceNameInputEditable}"
        @keyup=${(e: KeyboardEvent) => {onKeyUp(e);}}/>
      <button ?disabled="${!this.createButtonEnabled}" id="create"
        @click=${onClickCreate}>Create slice
      </button>
      <button ?disabled="${!this.deleteButtonEnabled}"  id="delete"
        @click=${onClickDelete}>Delete slice
      </button>
    `;
    // clang-format on
  }

  renderSliceSelector() {
    const selectedSliceName = this.sliceService.selectedSliceName;
    const sliceNames = [''].concat(this.sliceService.sliceNames);

    const handleSelectChange = (e: Event) => {
      const selectedValue = (e.target as HTMLSelectElement).value;
      this.selectSlice(selectedValue);
    };

    // clang-format off
    return html`
      <label class="dropdown-label">Selected slice</label>
      <select class="dropdown" id="slice-selector"
        .value=${selectedSliceName}
        @change=${handleSelectChange}>
        ${sliceNames.map(sliceName => {
          const isSelected = sliceName === selectedSliceName;
          return html`
            <option ?selected="${isSelected}" value=${sliceName}>
              ${sliceName}
            </option>`;
        })}
      </select>
    `;
    // clang-format on
  }

  /**
   * Create checkboxes for each value of each categorical feature.
   */
  renderFilters() {
    // Update the filterdict to match the checkboxes.
    const onChange = (e: Event, key: string) => {
      if ((e.target as HTMLInputElement).checked) {
        this.sliceByFeatures.push(key);
      } else {
        const index = this.sliceByFeatures.indexOf(key);
        this.sliceByFeatures.splice(index, 1);
      }
    };

    const renderFeatureCheckbox = (key: string) => {
      // clang-format off
      return html`
        <div>
          <div class='checkbox-holder'>
            <lit-checkbox
              ?checked=${this.sliceByFeatures.includes(key)}
              @change='${(e: Event) => {onChange(e, key);}}'
              label=${key}>
            </lit-checkbox>
          </div>
        </div>
      `;
      // clang-format on
    };
    return html`${
        this.groupService.categoricalAndNumericalFeatureNames.map(
            key => renderFeatureCheckbox(key))}`;
  }

  private renderNumSlices() {
    if (this.anyCheckboxChecked) {
      return html`
        <div class="dropdown-label">
          (${
          this.groupService.numIntersectionsLabel(this.sliceByFeatures)} slices)
        </div>
      `;
    }
    return '';
  }

  render() {
    return html`
      <div class="toolbar" id="slice-toolbar">
        ${this.renderSliceSelector()}
        ${this.renderButtons()}
      </div>
      <div class="toolbar" >
        <label class="dropdown-label">Slice by feature</label>
        ${this.renderFilters()}
        ${this.renderNumSlices()}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-slice-toolbar': SliceToolbar;
  }
}
