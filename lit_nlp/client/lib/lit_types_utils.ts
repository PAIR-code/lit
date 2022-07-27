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

// For consistency with types.ts.
// tslint:disable: enforce-name-casing

import {LitName, REGISTRY} from './lit_types';
import {LitMetadata, Spec} from './types';

/**
 * Creates and returns a new LitType instance.
 * @param typeName: The name of the desired LitType.
 * @param constructorParams: A dictionary of properties to set on the LitType.
 * For example, {'show_in_data_table': true}.
 */
export function createLitType(
    typeName: LitName, constructorParams: {[key: string]: unknown} = {}) {
  const litType = REGISTRY[typeName];
  // tslint:disable-next-line:no-any
  const newType = new (litType as any)();
  newType.__name__ = typeName;

  // Excluded properties are passed through in the Python serialization
  // of LitTypes and can be ignored by the frontend.
  const excluded = ['__mro__'];
  for (const key in constructorParams) {
    if (excluded.includes(key)) {
      continue;
    } else if (key in newType) {
      newType[key] = constructorParams[key];
    } else {
      throw new Error(
          `Attempted to set unrecognized property ${key} on ${newType}.`);
    }
  }

  return newType;
}


interface SerializedSpec {
  [key: string]: {__name__: string};
}

/**
 * Converts serialized LitTypes within a Spec into LitType instances.
 */
export function deserializeLitTypesInSpec(serializedSpec: SerializedSpec): Spec {
  const typedSpec: Spec = {};
  for (const key of Object.keys(serializedSpec)) {
    typedSpec[key] =
        createLitType(serializedSpec[key].__name__, serializedSpec[key] as {});
  }
  return typedSpec;
}


/**
 * Converts serialized LitTypes within the LitMetadata into LitType instances.
 */
export function deserializeLitTypesInLitMetadata(metadata: LitMetadata):
    LitMetadata {
  for (const model of Object.keys(metadata.models)) {
    metadata.models[model].spec.input =
        deserializeLitTypesInSpec(metadata.models[model].spec.input);
    metadata.models[model].spec.output =
        deserializeLitTypesInSpec(metadata.models[model].spec.output);
  }

  for (const dataset of Object.keys(metadata.datasets)) {
    metadata.datasets[dataset].spec =
        deserializeLitTypesInSpec(metadata.datasets[dataset].spec);
  }

  for (const generator of Object.keys(metadata.generators)) {
    metadata.generators[generator].configSpec =
        deserializeLitTypesInSpec(metadata.generators[generator].configSpec);
    metadata.generators[generator].metaSpec =
        deserializeLitTypesInSpec(metadata.generators[generator].metaSpec);
  }

  for (const interpreter of Object.keys(metadata.interpreters)) {
    metadata.interpreters[interpreter].configSpec = deserializeLitTypesInSpec(
        metadata.interpreters[interpreter].configSpec);
    metadata.interpreters[interpreter].metaSpec =
        deserializeLitTypesInSpec(metadata.interpreters[interpreter].metaSpec);
  }

  metadata.littypes = deserializeLitTypesInSpec(metadata.littypes);
  return metadata;
}
