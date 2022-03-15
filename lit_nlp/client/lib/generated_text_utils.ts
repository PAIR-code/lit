/**
 * @license
 * Copyright 2022 Google LLC
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
 * Shared helper functions for text generation / seq2seq models.
 */
import {GeneratedTextCandidate, IndexedInput, Input, LitName, Preds, Spec} from './types';
import {findSpecKeys, isLitSubtype} from './utils';

/**
 * Preds type for text generation.
 */
export interface GeneratedTextResult {
  [outputFieldName: string]: GeneratedTextCandidate[];
}

/**
 * Types for sequence generation output
 */
export const GENERATION_TYPES: LitName[] =
    ['GeneratedText', 'GeneratedTextCandidates'];

/**
 * Convert generation results, which may contain a mix of types, to a
 * canonical form.
 */
export function canonicalizeGenerationResults(
    result: Preds, outputSpec: Spec): GeneratedTextResult {
  const preds: GeneratedTextResult = {};
  for (const key of Object.keys(result)) {
    if (isLitSubtype(outputSpec[key], 'GeneratedText')) {
      preds[key] = [[result[key], null]];
    }
    if (isLitSubtype(outputSpec[key], 'GeneratedTextCandidates')) {
      preds[key] = result[key];
    }
  }
  return preds;
}

/**
 * Return texts corresponding to string or candidate list fields.
 */
export function getFlatTexts(
    inputReferenceKeys: string[], input?: Input|Preds|null): string[] {
  if (input == null) return [];

  const ret: string[] = [];
  for (const key of inputReferenceKeys) {
    const fieldData = input[key];
    // ReferenceTexts or GeneratedTextCandidates
    if (fieldData instanceof Array) {
      for (const textAndScore of fieldData) {
        ret.push(textAndScore[0]);
      }
    } else {
      ret.push(fieldData);
    }
  }
  return ret;
}

/**
 * Get all reference texts which are referenced in the model's output spec.
 */
export function getAllReferenceTexts(
    dataSpec: Spec, outputSpec: Spec, input?: IndexedInput|null): string[] {
  if (input == null) return [];

  // Search input fields: anything referenced in model's output spec
  const inputReferenceKeys = new Set<string>();
  for (const outKey of findSpecKeys(outputSpec, GENERATION_TYPES)) {
    const parent = outputSpec[outKey].parent;
    if (parent && dataSpec[parent]) {
      inputReferenceKeys.add(parent);
    }
  }
  return getFlatTexts([...inputReferenceKeys], input.data);
}

/**
 * Get all output texts from model predictions.
 */
export function getAllOutputTexts(
    outputSpec: Spec, preds?: GeneratedTextResult|null): string[] {
  return getFlatTexts(findSpecKeys(outputSpec, GENERATION_TYPES), preds);
}
