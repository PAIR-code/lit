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
 * Base class for LIT modules and other reactive elements.
 */

// tslint:disable:no-new-decorators
import {MobxLitElement} from '@adobe/lit-mobx';
import {IReactionDisposer, IReactionOptions, IReactionPublic, reaction} from 'mobx';

type ReactionInputFn<T> = (r: IReactionPublic) => T;

/**
 * Extension of MobxLitElement, with automatic reaction disposal.
 *
 * Prefer using this to MobxLitElement if your element has a finite lifetime,
 * since ReactiveElement will automatically clean up reactions when your element
 * is disconnected.
 * TODO(lit-dev): set reactions to be activated in connectedCallback() ?
 */
export abstract class ReactiveElement extends MobxLitElement {
  private readonly reactionDisposers: IReactionDisposer[] = [];

  override disconnectedCallback() {
    super.disconnectedCallback();
    this.reactionDisposers.forEach(disposer => {
      disposer();
    });
  }

  /**
   * A simple wrapper method around mobx `reaction`. Note that this reaction
   * is not called immediately (for that, use this.reactImmediately).
   * Automatically sets up the reaction for disposal upon the component being
   * disconnected.
   */
  protected react<T>(
      fn: ReactionInputFn<T>, effect: (arg: T, r: IReactionPublic) => void,
      opts: IReactionOptions = {}) {
    const disposer = reaction(fn, effect, opts);
    this.reactionDisposers.push(disposer);
  }

  /**
   * A simple wrapper method around mobx `reaction`, with the `fireImmediately`
   * option set to true. Automatically sets up the reaction for disposal upon
   * the component being disconnected.
   */
  protected reactImmediately<T>(
      fn: ReactionInputFn<T>, effect: (arg: T, r: IReactionPublic) => void,
      opts: IReactionOptions = {}) {
    this.react(fn, effect, {...opts, fireImmediately: true});
  }
}
