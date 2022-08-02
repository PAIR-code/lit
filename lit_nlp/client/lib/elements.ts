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
import {property} from 'lit/decorators';
import {IReactionDisposer, IReactionOptions, IReactionPublic, observable, reaction} from 'mobx';

type ReactionInputFn<T> = (r: IReactionPublic) => T;
// tslint:disable:no-any
type PendingReactionMap =
    Map<(arg: any, r: IReactionPublic) => void, {arg: any, r: IReactionPublic}>;
// tslint:enable:no-any

/**
 * Extension of MobxLitElement, with automatic reaction disposal.
 *
 * Prefer using this to MobxLitElement if your element has a finite lifetime,
 * since ReactiveElement will automatically clean up reactions when your element
 * is disconnected.
 * TODO(lit-dev): set reactions to be activated in connectedCallback() ?
 */
export abstract class ReactiveElement extends MobxLitElement {
  // TODO(b/204677206): remove this once we clean up property declarations.
  __allowInstanceProperties = true;  // tslint:disable-line

  /**
   * Indicates if the element should perform reactions. Set to 0 if the
   * element is not visible to the user and any other number if it is visible.
   * Not using a boolean here as web components cannot have boolean attributes
   * that default to true if not specified due to how boolean attributes are
   * passed to elements, given that we want this attribute to default to true
   * for backwards compabitibility with elements that don't have this value set
   * through an attribute.
   */
  @observable @property({type: Number}) shouldReact = 1;

  private readonly reactionDisposers: IReactionDisposer[] = [];

  // Map of pending reactions that are stored when an element is in a state
  // where it shouldn't react which are performed when the element is set to
  // react again.
  private readonly pendingReactions: PendingReactionMap = new Map();

  constructor() {
    super();
    reaction(() => this.shouldReact, shouldReact => {
      if (shouldReact === 0) {
        return;
      }
      this.performPendingReactions();
    });
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    this.reactionDisposers.forEach(disposer => {
      disposer();
    });
  }

  /**
   * Performs all pending reactions and clears them from the map.
   */
  private performPendingReactions() {
    this.pendingReactions.forEach((value, effect) => {
      effect.apply(window, [value.arg, value.r]);
    });
    this.pendingReactions.clear();
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
    // Wrapper function to pass to the mobx reaction call. If the element is
    // set to react, then the effect happens immediately. If not, the arguments
    // for the effect are stored in the map, keyed by the effect, so that
    // the latest reaction of each type is stored for later use.
    const reactWrapper = (arg: T, r: IReactionPublic) => {
      if (this.shouldReact !== 0) {
        effect.apply(window, [arg, r]);
      } else {
        this.pendingReactions.set(effect, {arg, r});
      }
    };
    const disposer = reaction(fn, reactWrapper, opts);
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
