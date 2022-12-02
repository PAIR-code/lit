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
import difflib from 'difflib';

import {ScoredTextCandidates} from './dtypes';
import {GeneratedText, GeneratedTextCandidates, LitTypeTypesList, LitTypeWithParent} from './lit_types';
import {IndexedInput, Input, Preds, Spec} from './types';
import {findSpecKeys} from './utils';

// tslint:disable-next-line:no-any difflib does not support Closure imports
// difflib declare placeholder - DO NOT REMOVE

/**
 * Preds type for text generation.
 */
export interface GeneratedTextResult {
  [outputFieldName: string]: ScoredTextCandidates;
}

/**
 * Types for sequence generation output
 */
export const GENERATION_TYPES: LitTypeTypesList =
    [GeneratedText, GeneratedTextCandidates];

/**
 * Convert generation results, which may contain a mix of types, to a
 * canonical form.
 */
export function canonicalizeGenerationResults(
    result: Preds, outputSpec: Spec): GeneratedTextResult {
  const preds: GeneratedTextResult = {};
  for (const key of Object.keys(result)) {
    if (outputSpec[key] instanceof GeneratedText) {
      preds[key] = [[result[key], null]];
    }
    if (outputSpec[key] instanceof GeneratedTextCandidates) {
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
    const {parent} = outputSpec[outKey] as LitTypeWithParent;
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

/**
 * Mode for diffs against reference text.
 */
export enum DiffMode {
  NONE = 'None',
  WORD = 'Word',
  CHAR = 'Character',
}

/**
 * Container type for a diff between two texts.
 */
export interface TextDiff {
  inputStrings: string[];
  outputStrings: string[];
  equal: boolean[];
}

/**
 * Uses difflib library to compute character differences between the input
 * strings and returns a TextDiff object, which contains arrays of parsed
 * segments from both strings and an array of booleans indicating whether the
 * corresponding change type is 'equal.'
 */
export function getTextDiff(
    targetText: string, outputText: string, byWord: boolean): TextDiff {
  // Use difflib library to compute opcodes, which contain a group of changes
  // between the two input strings. Each opcode contains the change type and
  // the start/end of the concerned characters/words in each string.
  const targetWords = targetText.split(' ');
  const outputWords = outputText.split(' ');

  const matcher = byWord ?
      new difflib.SequenceMatcher(() => false, targetWords, outputWords) :
      new difflib.SequenceMatcher(() => false, targetText, outputText);
  const opcodes = matcher.getOpcodes();

  // Store an array of the parsed segments from both strings and whether
  // the change type is 'equal.'
  const inputStrings: string[] = [];
  const outputStrings: string[] = [];
  const equal: boolean[] = [];

  for (const opcode of opcodes) {
    const changeType = opcode[0];
    const startA = Number(opcode[1]);
    const endA = Number(opcode[2]);
    const startB = Number(opcode[3]);
    const endB = Number(opcode[4]);

    equal.push((changeType === 'equal'));

    if (byWord) {
      inputStrings.push(targetWords.slice(startA, endA).join(' '));
      outputStrings.push(outputWords.slice(startB, endB).join(' '));
    } else {
      inputStrings.push(targetText.slice(startA, endA));
      outputStrings.push(outputText.slice(startB, endB));
    }
  }

  const textDiff: TextDiff = {inputStrings, outputStrings, equal};
  return textDiff;
}
