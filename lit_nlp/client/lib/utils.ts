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
 * Shared helper functions used across the app.
 */

import * as d3 from 'd3';  // Used for array helpers.

import {FacetMap, LitName, LitType, ModelsMap, Spec} from './types';

/**
 * Random integer in range [min, max), where min and max are integers
 * (behavior on floats is undefined).
 */
export function randInt(min: number, max: number) {
  return Math.floor(min + Math.random() * (max - min));
}

/**
 * Determines whether or not two sets are equal.
 */
export function setEquals<T>(setA: Set<T>, setB: Set<T>) {
  if (setA.size !== setB.size) return false;
  for (const a of setA) {
    if (!setB.has(a)) return false;
  }
  return true;
}

/**
 * Determines whether two arrays contain the same (unique) items.
 */
export function arrayContainsSame<T>(arrayA: T[], arrayB: T[]) {
  return setEquals(new Set<T>(arrayA), new Set<T>(arrayB));
}

/**
 * Check if a spec field (LitType) is an instance of one or more type names.
 * This is analogous to using isinstance(litType, typesToFind) in Python,
 * and relies on exporting the Python class hierarchy in the __mro__ field.
 */
export function isLitSubtype(litType: LitType, typesToFind: LitName|LitName[]) {
  // TODO(lit-dev): figure out why this is occasionally called on an invalid
  // spec. Likely due to skew between keys and specs in specific modules when
  // dataset is changed, but worth diagnosing to make sure this doesn't mask a
  // larger issue.
  if (litType == null) return false;

  if (typeof typesToFind === 'string') {
    typesToFind = [typesToFind];
  }
  for (const typeName of typesToFind) {
    if (litType.__mro__.includes(typeName)) {
      return true;
    }
  }
  return false;
}

/**
 * Find all keys from the spec which match any of the specified types.
 */
export function findSpecKeys(
    spec: Spec, typesToFind: LitName|LitName[]): string[] {
  if (typeof typesToFind === 'string') {
    typesToFind = [typesToFind];
  }
  return Object.keys(spec).filter(
      key => isLitSubtype(spec[key], typesToFind as LitName[]));
}

/**
 * Flattens a nested array by a single level.
 */
export function flatten<T>(arr: T[][]): T[] {
  return d3.merge(arr);
}

/**
 * Permutes an array.
 */
export function permute<T>(arr: T[], perm: number[]): T[] {
  const sorted: T[] = [];
  for (let i = 0; i < arr.length; i++) {
    sorted.push(arr[perm[i]]);
  }
  return sorted;
}

/**
 * Handler for a keystroke that checks if the key pressed was enter,
 * and if so, calls the callback.
 * @param e Original event
 * @param callback User defined callback method.
 */
export function handleEnterKey(e: KeyboardEvent, callback: () => void) {
  if (e.key === 'Enter') {
    callback();
  }
}

/**
 *  Converts the margin value to the threshold for binary classification.
 */
export function getThresholdFromMargin(margin: number) {
  if (margin == null) {
    return .5;
  }
  return margin === 0 ? .5 : 1 / (1 + Math.exp(-margin));
}

/**
 * Shortens the id of an input data to be displayed in the UI.
 */
export function shortenId(id: string|null) {
  if (id == null) {
    return;
  }
  return id.substring(0, 6);
}

/**
 * Return true for finite numbers.
 * Also coerces numbers in string form (e.g., "2")
 */
// tslint:disable-next-line:no-any
export function isNumber(num: any) {
  if (typeof num === 'number') {
    return num - num === 0;
  }
  if (typeof num === 'string' && num.trim() !== '') {
    return Number.isFinite(+num);
  }
  return false;
}

/**
 * Return an array of provided size with sequential numbers starting at 0.
 */
export function range(size: number) {
  return [...Array.from<number>({length: size}).keys()];
}

/**
 * Sum of the items in an array.
 */
export function sumArray(array: number[]) {
  return array.reduce((a, b) => a + b, 0);
}

/**
 * Cumulative sum for an array.
 */
export function cumSumArray(array: number[]) {
  const newArray: number[] = [];
  array.reduce((a, b, i) => newArray[i] = a + b, 0);
  return newArray;
}

/**
 * Python-style array comparison.
 * Compare on first element, then second, and so on until a mismatch is found.
 * If one array is a prefix of another, the longer one is treated as larger.
 * Example:
 *   [1] < [1,2] < [1,3] < [1,3,0] < [2]
 */
export function compareArrays(a: d3.Primitive[], b: d3.Primitive[]): number {
  // If either is empty, the longer one wins.
  if (a.length === 0 || b.length === 0) {
    return d3.ascending(a.length, b.length);
  }
  // If both non-empty, compare the first element.
  const firstComparison = d3.ascending(a[0], b[0]);
  if (firstComparison !== 0) {
    return firstComparison;
  }
  // If first element matches, recurse.
  return compareArrays(a.slice(1), b.slice(1));
}

/**
 * Checks if any of the model output specs contain any of the provided types.
 * Can be provided a single type string or a list of them.
 */
export function doesOutputSpecContain(
    models: ModelsMap, typesToCheck: LitName|LitName[]): boolean {
  const modelNames = Object.keys(models);
  for (let modelNum = 0; modelNum < modelNames.length; modelNum++) {
    const outputSpec = models[modelNames[modelNum]].spec.output;
    if (findSpecKeys(outputSpec, typesToCheck).length) {
      return true;
    }
  }
  return false;
}


/**
 * Helper function to make an object into a human readable key.
 * Sorts object keys, so order of object does not matter.
 */
export function objToDictKey(dict: FacetMap) {
  return Object.keys(dict).sort().map(key => `${key}:${dict[key]}`).join('/');
}

/**
 * Rounds a number up to the provided number of decimal places.
 */
export function roundToDecimalPlaces(num: number, places: number) {
  if (places < 0) {
    return num;
  }
  const numForPlaces = Math.pow(10, places);
  return Math.round((num + Number.EPSILON) * numForPlaces) / numForPlaces;
}

/**
 * Format for showing the user; may be a regression score or classification index.
 */
export function formatLabelNumber(num: number) {
  if (typeof num === 'number' && num % 1 !== 0) {
    return num.toFixed(3);
  }
  return num;
}