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

import {html, TemplateResult} from 'lit';
import {unsafeHTML} from 'lit/directives/unsafe-html.js';

import {marked} from 'marked';
import {FacetMap, LitName, LitType, ModelInfoMap, Spec} from './types';

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
  if (margin == null) {
    return .5;
  }
  return margin === 0 ? .5 : 1 / (1 + Math.exp(-margin));
}

/**
 *  Converts the threshold value for binary classification to the margin.
 */
export function getMarginFromThreshold(threshold: number) {
  const margin = threshold !== 1 ?
      (threshold !== 0 ? Math.log(threshold / (1 - threshold)) : -5) :
      5;
  return margin;
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
    models: ModelInfoMap, typesToCheck: LitName|LitName[],
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
    models: ModelInfoMap, typesToCheck: LitName|LitName[],
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
    const predictionLabels = litType.vocab!;
    const nullIdx = litType.null_idx;
    return predictionLabels.length === 2 && nullIdx != null;
}

/** Returns if a LitType has a parent field. */
export function hasParent(litType: LitType) {
    return litType.parent != null;
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
 * Copies a value to the user's clipboard.
 */
export function copyToClipboard(value: string) {
  const tempInput = document.createElement("input");
  tempInput.value = value;
  document.body.appendChild(tempInput);
  tempInput.select();
  document.execCommand("copy");
  document.body.removeChild(tempInput);
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
  const chunkWord = (word: string) => {
    const maxLen = 15;
    const chunks: string[] = [];
    for (let i=0; i<word.length; i+=maxLen) {
      chunks.push(word.slice(i, i+maxLen));
    }
    // This is not an empty string, it is a non-printing space.
    const zeroWidthSpace = '​';
    return chunks.join(zeroWidthSpace);
  };
  return sent.split(' ').map(word => chunkWord(word)).join(' ');
}

/**
 * Converts any URLs into clickable links.
 * TODO(lit-dev): write unit tests for this.
 */
export function linkifyUrls(
    text: string,
    target: '_self'|'_blank'|'_parent'|'_top' = '_self'): TemplateResult {
  const ret: Array<string|TemplateResult> = [];  // return segments
  let lastIndex = 0;  // index of last character added to return segments
  // Find urls and make them real links.
  // Similar to gmail and other apps, this assumes terminal punctuation is
  // not part of the url.
  const matcher = /https?:\/\/[^\s]+[^.?!\s]/g;
  const formatLink = (url: string) =>
      html`<a href=${url} target=${target}>${url}</a>`;
  for (const match of text.matchAll(matcher)) {
    ret.push(text.slice(lastIndex, match.index));
    lastIndex = match.index! + match[0].length;
    ret.push(formatLink(text.slice(match.index, lastIndex)));
  }
  ret.push(text.slice(lastIndex, text.length));
  return html`${ret}`;
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
  return range > 100 ? 10: range > 10 ? 1 : range > 1 ? 0.1 : 0.01;
}

/** Convert a markdown string into an HTML template for rendering. */
export function getTemplateStringFromMarkdown(markdown: string) {
  const htmlStr = marked(markdown);
  return unsafeHTML(htmlStr);
}
