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

type LitClass = 'LitType';

/**
 * Data classes used in configuring front-end components to describe
 * input data and model outputs.
 */
export class LitType {
  // tslint:disable:enforce-name-casing
  __class__: LitClass|'type' = 'LitType';
  // TODO(b/162269499): Replace this with LitName, when created.
  // tslint:disable-next-line:no-any
  __name__: any|undefined;
  // TODO(b/162269499): __mro__ is included here to temporarily ensure
  // type equivalence betwen the old `LitType` and new `LitType`.
  __mro__: string[] = [];
  readonly required: boolean = true;
  // TODO(b/162269499): Replace this with `unknown` after migration.
  // tslint:disable-next-line:no-any
  readonly default: any|undefined = null;
  // TODO(b/162269499): Update to camel case once we've replaced old LitType.
  show_in_data_table: boolean = false;

  // TODO(b/162269499): Add isCompatible functionality.
}

/**
 * A string LitType.
 */
export class String extends LitType {
  override default: string = '';
}

/**
 * A scalar value, either a single float or int.
 */
export class Scalar extends LitType {
  override default: number = 0;
  min_val: number = 0;
  max_val: number = 1;
  step: number = .01;
}
