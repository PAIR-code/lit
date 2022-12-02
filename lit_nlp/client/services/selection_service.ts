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
import {action, computed, observable} from 'mobx';

import {IndexedInput, ServiceUser} from '../lib/types';

import {LitService} from './lit_service';
import {SelectionObservedByUrlService} from './url_service';

/**
 * The AppState interface for working with the SelectionService
 */
export interface AppState {
  getIndexById: (ids: string|null) => number;
  currentInputData: IndexedInput[];
  getCurrentInputDataById: (id: string) => IndexedInput | null;
  getExamplesById: (ids: string[]) => IndexedInput[];
}

/**
 * A singleton service for managing App selections of input data.
 */
export class SelectionService extends LitService implements
    SelectionObservedByUrlService {
  constructor(private readonly appState: AppState) {
    super();
  }

  @observable private readonly selectedIdsSet = new Set<string>();
  @observable private primarySelectedIdInternal: string|null = null;

  // Track the last user, so components can avoid resetting on selections they
  // triggered.
  @observable private lastUserInternal?: ServiceUser;


  // Tracks the last updated data indices to calculate selection.
  @observable private shiftSelectionStartIndexInternal: number = 0;
  @observable private shiftSelectionEndIndexInternal: number = 0;

  @computed
  get lastUser() {
    return this.lastUserInternal;
  }

  @computed
  get primarySelectedId() {
    return this.primarySelectedIdInternal;
  }

  @computed
  get selectedIds(): string[] {
    return [...this.selectedIdsSet.values()].sort(
        (a, b) =>
            this.appState.getIndexById(a) - this.appState.getIndexById(b));
  }

  @computed
  get selectedInputData(): IndexedInput[] {
    if (this.selectedIds.length === 0) {
      return [];
    }
    return this.appState.currentInputData.filter((inputData) => {
      return this.selectedIdsSet.has(inputData.id);
    });
  }

  @computed
  get primarySelectedInputData(): IndexedInput|null {
    if (this.primarySelectedId !== null) {
      return this.appState.getCurrentInputDataById(this.primarySelectedId);
    }
    return null;
  }

  @computed
  get shiftSelectionStartIndex() {
    return this.shiftSelectionStartIndexInternal;
  }

  @computed
  get shiftSelectionEndIndex() {
    return this.shiftSelectionEndIndexInternal;
  }

  @action
  setLastUser(user?: ServiceUser) {
    this.lastUserInternal = user;
  }

  @action
  setPrimarySelection(id: string|null, user?: ServiceUser) {
    if (id == null || this.selectedIdsSet.has(id)) {
      // Primary id is within the selected set, or we're clearing selection.
      this.primarySelectedIdInternal = id;
      this.setLastUser(user);
    } else {
      // Not in main selection, so update that.
      this.selectIds([id], user);
    }
  }

  @action
  setShiftSelectionSpan(startIndex: number, endIndex: number) {
    this.shiftSelectionStartIndexInternal = startIndex;
    this.shiftSelectionEndIndexInternal = endIndex;
  }

  @action
  selectIds(ids: string[], user?: ServiceUser) {
    this.selectedIdsSet.clear();
    ids.forEach(id => {
      this.selectedIdsSet.add(id);
    });
    if (this.primarySelectedId !== null &&
        this.selectedIdsSet.has(this.primarySelectedId)) {
      // Don't change primary selected id.
    } else if (this.selectedIdsSet.size === 0) {
      // Clear primary if selection is empty.
      this.primarySelectedIdInternal = null;
    } else {
      // Default to first point.
      this.primarySelectedIdInternal = ids[0];
    }
    this.setLastUser(user);

    if (ids.length === 1) {
      const index = this.appState.getIndexById(ids[0]);
      this.setShiftSelectionSpan(index, index);
    }
  }

  @action
  selectAll(user?: ServiceUser) {
    const ids = this.appState.currentInputData.map(d => d.id);
    this.selectIds(ids, user);
  }

  /**
   * Sync state from another selection service.
   * Atomic, so we only trigger a single set of reactions.
   */
  @action
  syncFrom(other: SelectionService) {
    this.selectIds(other.selectedIds);
    this.setPrimarySelection(other.primarySelectedId);
  }

  isIdSelected(id: string) {
    return this.selectedIdsSet.has(id);
  }

  /**
   * Gets the selected input data if any selected, otherwise all input data.
   */
  @computed
  get selectedOrAllInputData(): IndexedInput[] {
    return this.selectedInputData.length > 0 ? this.selectedInputData :
                                               this.appState.currentInputData;
  }
}
