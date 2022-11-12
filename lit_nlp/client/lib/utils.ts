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

// For consistency with types.ts.
// tslint:disable: enforce-name-casing

import * as d3 from 'd3';  // Used for array helpers.

import {unsafeHTML} from 'lit/directives/unsafe-html.js';

import {marked} from 'marked';
import {LitName, LitType, LitTypeTypesList, LitTypeWithParent, MulticlassPreds, LIT_TYPES_REGISTRY} from './lit_types';
import {FacetMap, LitMetadata, ModelInfoMap, SerializedLitMetadata, SerializedSpec, Spec} from './types';

/** Calculates the mean for a list of numbers */
export function mean(values: number[]): number {
  return values.reduce((a, b) => a + b) / values.length;
}

/** Calculates the median for a list of numbers. */
export function median(values: number[]): number {
  const sorted = [...values].sort();
  const medIdx = Math.floor(sorted.length / 2);
  let median: number;

  if (sorted.length % 2 === 0) {
    const upper = sorted[medIdx];
    const lower = sorted[medIdx - 1];
    median = (upper + lower) / 2;
  } else {
    median = sorted[medIdx];
  }

  return median;
}

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

/** Determines whether two maps contain the same keys and values. */
export function mapsContainSame<T>(mapA: Map<string, T>, mapB: Map<string, T>) {
  const mapAKeys = Array.from(mapA.keys());
  if (!arrayContainsSame(mapAKeys, Array.from(mapB.keys()))) {
    return false;
  }
  for (const key of mapAKeys) {
    if (mapA.get(key) !== mapB.get(key)) {
      return false;
    }
  }
  return true;
}

/** Returns a list of names corresponding to LitTypes. */
export function getTypeNames(litTypes: LitTypeTypesList) : LitName[] {
  // TypeScript treats `typeof LitType` as a constructor function.
  // Cast to any to access the name property.
  // tslint:disable-next-line:no-any
  return litTypes.map(t => (t as any).name);
}

/** Returns a list of LitTypes corresponding to LitNames. */
// We return the equivalent of LitTypeTypesList, but TypeScript constructor
// functions do not have the same signature as the types themselves.
// tslint:disable-next-line:no-any
export function getTypes(litNames: LitName|LitName[]) : any {
  if (typeof litNames === 'string') {
    litNames = [litNames];
  }

  return litNames.map(litName => LIT_TYPES_REGISTRY[litName]);
}

/**
 * Creates and returns a new LitType instance.
 * @param litTypeConstructor: A constructor for the LitType instance.
 * @param constructorParams: A dictionary of properties to set on the LitType.
 * For example, {'show_in_data_table': true}.
 *
 * We use this helper instead of directly creating a new T(), because this
 * allows creation of LitTypes dynamically from the metadata returned from the
 * server via the `/get_info` API, and allows updating class properties on
 * creation time.
 */
export function createLitType<T>(
    litTypeConstructor: new () => T,
    constructorParams: {[key: string]: unknown} = {}): T {
  const litType = new litTypeConstructor();

  // Temporarily make this LitType generic to set properties dynamically.
  // tslint:disable-next-line:no-any
  const genericLitType = litType as any;

  for (const key in constructorParams) {
    if (key in genericLitType) {
      genericLitType[key] = constructorParams[key];
    } else if (key !== '__name__') {  // Ignore __name__ property.
      throw new Error(`Attempted to set unrecognized property ${key} on ${
          genericLitType.name}.`);
    }
  }

  return genericLitType as T;
}

/**
 * Converts serialized LitTypes within a Spec into LitType instances.
 */
export function deserializeLitTypesInSpec(serializedSpec: SerializedSpec):
    Spec {
  const typedSpec: Spec = {};
  for (const key of Object.keys(serializedSpec)) {
    typedSpec[key] = createLitType(
        LIT_TYPES_REGISTRY[serializedSpec[key].__name__],
        serializedSpec[key] as {});
  }
  return typedSpec;
}

/**
 * Returns a deep copy of the given spec.
 */
export function cloneSpec(spec: Spec): Spec {
  const newSpec: Spec = {};
  for (const [key, fieldSpec] of Object.entries(spec)) {
    newSpec[key] =
        createLitType(LIT_TYPES_REGISTRY[fieldSpec.name], fieldSpec as {});
  }
  return newSpec;
}

/**
 * Converts serialized LitTypes within the LitMetadata into LitType instances.
 */
export function deserializeLitTypesInLitMetadata(
    metadata: SerializedLitMetadata): LitMetadata {
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

  return metadata;
}

type CandidateLitTypeTypesList = (typeof LitType)|LitTypeTypesList;

function wrapSingletonToList<Type>(candidate: Type|Type[]):
    Type[] {
  if (!Array.isArray(candidate)) {
    candidate = [candidate];
  }

  return candidate;
}

/**
 * Returns whether the litType is a subtype of any of the typesToFind.
 * @param litType: The LitType to check.
 * @param typesToFind: Either a single or list of parent LitType candidates.
 */
export function isLitSubtype(
    litType: LitType, typesToFind: CandidateLitTypeTypesList) {
  if (litType == null) return false;

  const typesToFindList = wrapSingletonToList(typesToFind);
  for (const typeName of typesToFindList) {
    if (litType instanceof typeName) {
      return true;
    }
  }
  return false;
}


/**
 * Returns all keys in the given spec that are subtypes of the typesToFind.
 * @param spec: A Spec object.
 * @param typesToFind: Either a single or list of parent LitType candidates.
 */
export function findSpecKeys(
    spec: Spec, typesToFind: CandidateLitTypeTypesList): string[] {
  const typesToFindList = wrapSingletonToList(typesToFind);
  return Object.keys(spec).filter(
      key => isLitSubtype(spec[key], typesToFindList));
}


/**
 * Return a new object with the selected keys from the old one.
 */
export function filterToKeys<V>(obj: {[key: string]: V}, keys: string[]) {
  const ret: {[key: string]: V} = {};
  for (const key of keys) {
    ret[key] = obj[key];
  }
  return ret;
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
  return !margin ? .5 : 1 / (1 + Math.exp(-margin));
}

/**
 *  Converts the threshold value for binary classification to the margin.
 */
export function getMarginFromThreshold(threshold: number) {
  return threshold === 1 ?  5 :
         threshold === 0 ? -5 : Math.log(threshold / (1 - threshold));
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
    models: ModelInfoMap, typesToCheck: CandidateLitTypeTypesList,
    extraCheck?: (litType: LitType) => boolean): boolean {
  const modelNames = Object.keys(models);
  for (let modelNum = 0; modelNum < modelNames.length; modelNum++) {
    const outputSpec = models[modelNames[modelNum]].spec.output;
    const matchedSpecs = findSpecKeys(outputSpec, typesToCheck);
    // If there are matching fields, and there is no extra check then return
    // true, otherwise return true if the extra check suceeds for any field.
    if (matchedSpecs.length) {
      if (extraCheck != null) {
        for (const matchedSpec of matchedSpecs) {
          if (extraCheck(outputSpec[matchedSpec])) {
            return true;
          }
        }
      } else {
        return true;
      }
    }
  }
  return false;
}

/**
 * Checks if any of the model input specs contain any of the provided types.
 * Can be provided a single type string or a list of them.
 */
export function doesInputSpecContain(
    models: ModelInfoMap, typesToCheck: CandidateLitTypeTypesList,
    checkRequired: boolean): boolean {
  const modelNames = Object.keys(models);
  for (let modelNum = 0; modelNum < modelNames.length; modelNum++) {
    const inputSpec = models[modelNames[modelNum]].spec.input;
    let keys = findSpecKeys(inputSpec, typesToCheck);
    if (checkRequired) {
      keys = keys.filter(spec => inputSpec[spec].required);
    }
    if (keys.length) {
      return true;
    }
  }
  return false;
}

/** Returns if a LitType specifies binary classification. */
export function isBinaryClassification(litType: LitType) {
  if (litType instanceof MulticlassPreds) {
    const {vocab, null_idx: nullIdx}  = litType;
    return vocab.length === 2 && nullIdx != null;
  }

  return false;
}

/** Returns if a LitType has a parent field. */
export function hasParent(litType: LitType) {
    return (litType as LitTypeWithParent).parent != null;
}

/**
 * Helper function to make an object into a human readable key.
 * Sorts object keys, so order of object does not matter.
 */
export function facetMapToDictKey(dict: FacetMap) {
  return Object.keys(dict).sort().map(
      key => `${key}:${dict[key].displayVal}`).join(' ');
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
 * Processes a sentence so that no word exceeds a certain length by
 * chunking a long word into shorter pieces. This is useful when rendering
 * a table-- normally a table will stretch to fit the entire word length
 * (https://www.w3schools.com/cssref/pr_tab_table-layout.asp).
 * TODO(lit-dev): find a more long-term solution to this, since adding a
 * NPWS will make copy/pasting from the table behave strangely.
 */
export function chunkWords(sent: string) {
  function chunkWord (word: string) {
    const maxLen = 15;
    const chunks: string[] = [];
    for (let i=0; i<word.length; i+=maxLen) {
      chunks.push(word.slice(i, i+maxLen));
    }
    // This is not an empty string, it is a non-printing space.
    const zeroWidthSpace = '​';
    return chunks.join(zeroWidthSpace);
  }
  return sent.split(' ').map(word => chunkWord(word)).join(' ');
}

const CANVAS = document.createElement('canvas');
/**
 * Computes the width of a string given a CSS font specifier. If the
 * browser doesn't support <canvas> elements, the width will be computed
 * using the specified default character width.
 */
export function getTextWidth(text: string, font: string, defaultCharWidth: number): number {
  const context = CANVAS.getContext == null ? null : CANVAS.getContext('2d');
  if (context == null) {
    return text.length * defaultCharWidth;
  }
  context.font = font;
  const metrics = context.measureText(text);
  return metrics.width;
}

/**
 * Gets the offset to the beginning of each token in a sentence using
 * the specified token widths and space width.
 */
export function getTokOffsets(tokWidths: number[], spaceWidth: number): number[] {
  const tokOffsets: number[] = [];
  let curOffset = 0;
  for (let i = 0; i < tokWidths.length; i++) {
    tokOffsets.push(curOffset);
    curOffset += tokWidths[i] + spaceWidth;
  }
  return tokOffsets;
}

/** Creates a hash code from a string similar to Java's hashCode method. */
export function hashCode(str: string) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const chr = str.charCodeAt(i);
    hash  = ((hash << 5) - hash) + chr;
    hash |= 0; // Convert to 32bit integer
  }
  return hash;
}

/** Find all matching indices of the val in the provided arr. */
export function findMatchingIndices(arr: unknown[], val: unknown): number[] {
  const indices: number[] = [];
  for(let i = 0; i < arr.length; i++) {
    if (arr[i] === val) {
        indices.push(i);
    }
  }
  return indices;
}

/** Return new string with the Nth instance of orig replaced. */
export function replaceNth(str: string, orig: string, replacement: string,
                           n: number) {
  const escapedOrig = orig.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&');
  return str.replace(
      RegExp("^(?:.*?" + escapedOrig + "){" + n.toString() + "}"),
      x => x.replace(RegExp(escapedOrig + "$"), replacement));
}

/** Return a good step size given a range of values. */
export function getStepSizeGivenRange(range: number) {
  // Returns 0.1 for values of at least 1 and less than 10.
  // Returns 1 for values of at least 10 and less than 100.
  // Returns 10 for values of at least 100 and less than 1000.
  // And so on, both for larger ranges and smaller.
  return Math.pow(10, Math.floor(Math.log10(range)) - 1);
}

/** Convert a markdown string into an HTML template for rendering. */
export function getTemplateStringFromMarkdown(markdown: string) {

  // Render Markdown with link target _blank
  // See https://github.com/markedjs/marked/issues/144
  // and https://github.com/markedjs/marked/issues/655
  const renderer = new marked.Renderer();
  renderer.link = (href, title, text) => {
    const linkHtml =
        marked.Renderer.prototype.link.call(renderer, href, title, text);
    return linkHtml.replace('<a', '<a target=\'_blank\' ');
  };
  const htmlStr = marked(markdown, {renderer});

  return unsafeHTML(htmlStr);
}

/**
 * Convert a number range from strings to a function to check inclusion.
 *
 * Can use commas or spaces to separate individual numbers/ranges. Logic can
 * handle negative numbers and decimals. Ranges are inclusive.
 * e.x. "1, 2, 4-6" will match the numbers 1, 2, and numbers between 4 and 6.
 * e.x. "-.5-1.5 10" will match numbers between -.5 and 1.5 and also 10.
 */
export function numberRangeFnFromString(str: string): (num: number) => boolean {
  // Matches single numbers, including decimals with and without leading zeros,
  // and negative numbers. Also matches ranges of numbers separated by a hyphen.
  const regexStr = /(-?\d*(?:\.\d+)?)(?:-(-?\d*(?:\.\d+)?))?/g;

  // Convert the string into a list of ranges of numbers to match.
  const ranges: Array<[number, number]> = [];
  for (const [, beginStr, endStr] of str.matchAll(regexStr)) {
    if (beginStr.length === 0) {
      continue;
    }
    ranges.push([beginStr, endStr || beginStr].map(Number) as [number, number]);
  }

  // Returns a function that matches numbers against the ranges.
  return (num: number) => {
    if (ranges.length === 0) {
      return true;
    }
    for (const range of ranges) {
      if (num >= range[0] && num <= range[1]) {
        return true;
      }
    }
    return false;
  };
}

/** Return evenly spaced numbers between minValue and maxValue. */
export function linearSpace(
    minValue: number, maxValue: number, numSteps: number): number[] {
  if (minValue > maxValue) {
    return [];
  }

  const values = [];
  const step = (maxValue - minValue) / (numSteps - 1);
  for (let i = 0; i < numSteps; i++) {
    values.push(minValue + i * step);
  }
  return values;
}
