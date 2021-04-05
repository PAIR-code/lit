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
import {action, computed, observable, reaction} from 'mobx';

import {IndexedInput, ServiceUser} from '../lib/types';
import {arrayContainsSame} from '../lib/utils';

import {LitService} from './lit_service';
import {AppState, SelectionService} from './services';

/**
 * The name of the slice containing user-favorited items.
 */
export const STARRED_SLICE_NAME = 'Starred';

type SliceName = string;
type Id = string;

/**
 * A singleton service for managing App selections of input data.
 */
export class SliceService extends LitService {
  constructor(
      private readonly selectionService: SelectionService,
      private readonly appState: AppState) {
    super();

    reaction(() => selectionService.selectedInputData, selectedInputData => {
      // If selection doesn't match a slice, then reset selected slice to
      // no slice.
      if (this.selectedSliceName == null) return;
      if (this.namedSlices.has(this.selectedSliceName) &&
          !arrayContainsSame(
              selectedInputData.map(input => input.id),
              this.getSliceByName(this.selectedSliceName)!)) {
        this.setSelectedSliceName(null);
      }
    });
  }

  // Initialize with an empty slice to hold favorited items.
  @observable
  namedSlices =
      new Map<SliceName, Set<Id>>([[STARRED_SLICE_NAME, new Set<Id>()]]);
  @observable private selectedSliceNameInternal: string|null = null;

  @action
  setSelectedSliceName(name: string|null) {
    this.selectedSliceNameInternal = name;
  }

  @computed
  get selectedSliceName(): string|null {
    return this.selectedSliceNameInternal;
  }

  @action
  selectNamedSlice(name: string|null, user: ServiceUser = null) {
    if (name === null) {
      this.setSelectedSliceName(name);
      this.selectionService.selectIds([], user);
      return;
    }
    const sliceIds = this.getSliceByName(name);
    // Do nothing if slice name is not found.
    if (sliceIds === null) {
      return;
    }
    this.selectionService.selectIds(sliceIds, user);
    this.setSelectedSliceName(name);
  }

  @action
  addNamedSlice(name: SliceName, ids: string[]) {
    this.namedSlices.set(name, new Set<Id>(ids));
  }

  @action
  addIdsToSlice(name: SliceName, ids: string[]) {
    const sliceIds = this.namedSlices.get(name);
    if (sliceIds == null) {
      return;
    }

    ids.forEach((id) => {
      sliceIds.add(id);
    });
  }

  @action
  removeIdsFromSlice(name: SliceName, ids: string[]) {
    const sliceIds = this.namedSlices.get(name);
    if (sliceIds == null) {
      return;
    }

    ids.forEach((id) => {
      sliceIds.delete(id);
    });
  }

  @action
  deleteNamedSlice(name: SliceName) {
    this.namedSlices.delete(name);
  }

  @computed
  get sliceNames(): SliceName[] {
    return [...this.namedSlices.keys()];
  }

  getSliceByName(sliceName: SliceName): string[]|null {
    const sliceIds = this.namedSlices.get(sliceName);
    return sliceIds ? [...sliceIds] : null;
  }


  getSliceDataByName(sliceName: SliceName): IndexedInput[] {
    const ids = this.getSliceByName(sliceName);
    return this.appState.getExamplesById(ids!);
  }

  isInSlice(sliceName: SliceName, id: Id): boolean {
    const slice = this.namedSlices.get(sliceName);
    return slice ? slice.has(id) : false;
  }

  isSliceEmpty(sliceName: SliceName): boolean {
    const slice = this.getSliceByName(sliceName);
    if (slice == null) return true;

    return slice.length === 0;
  }

  areAllSlicesEmpty(): boolean {
    for (const slice of this.sliceNames) {
      if (!this.isSliceEmpty(slice)) {
        return false;
      }
    }
    return true;
  }
}
