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
import {hashCode} from '../lib/utils';
import {LitService} from './lit_service';

/**
 * Contains a status bar message and full error message for display in a modal.
 */
interface ErrorMessageHolder {
  message: string;
  fullMessage: string;
}

/**
 * A singleton class that handles all API loading status messages.
 */
export class StatusService extends LitService {
  private loadingId: number = 0;

  /**
   * An observable map of loading messages by loading id.
   */
  @observable private readonly loadingEvents = new Map<number, string>();
  @observable private readonly errorEvents =
      new Map<number, ErrorMessageHolder>();

  @computed
  get loadingMessages() {
    return [...this.loadingEvents.values()];
  }

  @computed
  get errorMessages() {
    return Array.from(this.errorEvents.values(), holder => holder.message);
  }

  @computed
  get errorFullMessages() {
    return Array.from(this.errorEvents.values(), holder => holder.fullMessage);
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
    if (!this.hasError) {
      return null;
    }
    let message = this.errorMessages[0];
    const otherCount = this.errorMessages.length - 1;
    if (otherCount > 0) {
      message +=
          ` (and ${otherCount} other error${otherCount > 1 ? 's)' : ')'}`;
    }
    return message;
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

  /**
   * Add an error to the LIT status.
   * @param message A short message to display in the status bar.
   * @param fullMessage Optional longer message to display in the details modal.
   * @param errorIdStr A string to ID this error by, for deduping and removing
   *                   of errors. If unspecified, then the message is used.
   */
  addError(message: string, fullMessage?: string, errorIdStr?: string) {
    const strToHash = errorIdStr != null ? errorIdStr : message;
    const id = hashCode(strToHash);
    if (fullMessage == null) {
      fullMessage = message;
    }
    this.errorEvents.set(id, {message, fullMessage});
  }

  /**
   * Removes an error from the LIT status.
   * @param errorIdStr A string to ID this error by, for removing any current
   *                   error from the LIT status.
   */
  removeError(errorIdStr: string) {
    this.errorEvents.delete(hashCode(errorIdStr));
  }

  clearErrors() {
    this.errorEvents.clear();
  }
}
