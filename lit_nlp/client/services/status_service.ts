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
import {computed, observable} from 'mobx';
import {LitService} from './lit_service';

/**
 * A singleton class that handles all API loading status messages.
 */
export class StatusService extends LitService {
  private loadingId: number = 0;
  private errorId: number = 0;
  /**
   * An observable map of loading messages by loading id.
   */
  @observable private readonly loadingEvents = new Map<number, string>();
  @observable private readonly errorEvents = new Map<number, string>();

  @computed
  get loadingMessages() {
    return [...this.loadingEvents.values()];
  }

  @computed
  get errorMessages() {
    return [...this.errorEvents.values()];
  }

  @computed
  get hasMessage() {
    return this.isLoading || this.hasError;
  }

  @computed
  get isLoading() {
    return this.loadingMessages.length > 0;
  }

  @computed
  get hasError() {
    return this.errorMessages.length > 0;
  }

  @computed
  get errorMessage() {
    const otherErrorCount = this.errorMessages.length - 1;
    return this.hasError ? `${this.errorMessages[0]} (${
                               otherErrorCount > 0 ?
                                   `and ${otherErrorCount} other error${
                                       otherErrorCount > 1 ? 's' : ''}, ` :
                                   ''}see console for more details)` :
                           '';
  }

  @computed
  get loadingMessage() {
    return this.isLoading ? this.loadingMessages[0] : '';
  }

  startLoading(message: string) {
    const id = this.loadingId;
    this.loadingEvents.set(id, message);
    this.loadingId += 1;
    return () => {
      this.endLoading(id);
    };
  }

  private endLoading(id: number) {
    this.loadingEvents.delete(id);
  }

  addError(message: string) {
    const id = this.errorId;
    this.errorEvents.set(id, message);
    this.errorId += 1;
  }

  clearErrors() {
    this.errorEvents.clear();
  }
}
