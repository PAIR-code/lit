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

/**
 * FocusService deals with the logic handling UI focus inside LIT, such as
 * on hover events.
 */

// tslint:disable:no-new-decorators
import {computed, observable, reaction} from 'mobx';
import {LitService} from './lit_service';
import {SelectionService} from './selection_service';

type IoType = "input"|"output";

/**
 * Describes what object is focused in the UI.
 */
export interface FocusData {
  datapointId: string;   // example id, within current selection
  io?: IoType;           // input means data, output means model preds
  fieldName?: string;    // key into data.spec() or model.output_spec()
  // tslint:disable-next-line:no-any
  subField?: any;        // pointer within the field, such as a token index
}

/**
 * A singleton class that handles focus.
 */
export class FocusService extends LitService {
  @observable private focusDataInternal: FocusData|null = null;

  /**
   * Gets the current focus data, or null if nothing is focused.
   */
  @computed
  get focusData(): FocusData|null {
    return this.focusDataInternal;
  }

  constructor(selectionService: SelectionService) {
    super();
    // If the primary selected input changes, reset the focus data.
    reaction(() => selectionService.primarySelectedInputData, selectedInput => {
      this.clearFocus();
    });
  }

  /**
   * Sets a datapoint to be focused.
   */
  setFocusedDatapoint(datapointId: string) {
    this.focusDataInternal = {datapointId};
  }

  /**
   * Sets a field within a datapoint to be focused.
   */
  setFocusedField(
      // tslint:disable-next-line:no-any
      datapointId: string, io: IoType, fieldName: string, subField?: any) {
    this.focusDataInternal = {datapointId, io, fieldName, subField};
  }

  /**
   * Clears focus so that nothing is focused.
   */
  clearFocus() {
    this.focusDataInternal = null;
  }
}
