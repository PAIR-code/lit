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

// tslint:disable:no-new-decorators class-name

/**
 * A dictionary of registered LitType names mapped to their constructor.
 * LitTypes are added using the @registered decorator.
 */
export const REGISTRY: {[litType: string]: LitType} = {};
// tslint:disable-next-line:no-any
function registered(target: any) {
  REGISTRY[target.name] = target;
}

const registryKeys = Object.keys(REGISTRY) as ReadonlyArray<string>;
/**
 * The types of all LitTypes in the registry, e.g.
 * 'String' | 'TextSegment' ...
 */
export type LitName = typeof registryKeys[number];

type LitClass = 'LitType';
type ScoredTextCandidates = Array<[text: string, score: number|null]>;

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
@registered
export class String extends LitType {
  override default: string = '';
}

/**
 * Text input (untokenized), a single string.
 */
@registered
export class TextSegment extends String {}

/**
 * An image, an encoded base64 ascii string (starts with 'data:image...').
 */
@registered
export class ImageBytes extends LitType {}


/**
 * Generated (untokenized) text.
 */
@registered
export class GeneratedText extends TextSegment {
  parent?: string = undefined;
}

/**
 * A list type.
 */
class _List extends LitType {
  override default: unknown[] = [];
}

/**
 * A list of (text, score) tuples.
 */
class StringCandidateList extends _List {
  override default: ScoredTextCandidates = [];
}

/**
 * Multiple candidates for GeneratedText.
 */
@registered
export class GeneratedTextCandidates extends StringCandidateList {
  /** Name of a TextSegment field to evaluate against. */
  parent?: string = undefined;
}

/**
 * Multiple candidates for TextSegment.
 */
@registered
export class ReferenceTexts extends StringCandidateList {}

/**
 * Multiple tokens with weight.
 */
@registered
export class TopTokens extends StringCandidateList {}


/**
 * TextSegment that should be interpreted as a URL.
 */
@registered
export class URL extends TextSegment {}

/**
 * A URL that was generated as part of a model prediction.
 */
@registered
export class GeneratedURL extends TextSegment {
  align?: string = undefined;  // Name of a field in the model output.
}

/**
 * TextSegment that should be interpreted as a search query.
 */
@registered
export class SearchQuery extends TextSegment {}

/**
 * A list of strings.
 */
@registered
export class StringList extends _List {
  override default: string[] = [];
}

/**
 * Tokenized text.
 */
@registered
export class Tokens extends StringList {
  /** Name of a TextSegment field from the input. */
  parent?: string = undefined;
  /** Optional mask token for input. */
  mask_token?: string = undefined;
  /** Optional prefix used in tokens. */
  token_prefix?: string = '##';
}

/**
 * Predicted tokens, as from a language model.
 * The inner list should contain (word, probability) in descending order.
 */
@registered
export class TokenTopKPreds extends _List {
  override default: ScoredTextCandidates[] = [];
  align?: string = undefined;
  parent?: string = undefined;
}

/**
 * A scalar value, either a single float or int.
 */
@registered
export class Scalar extends LitType {
  override default: number = 0;
  min_val: number = 0;
  max_val: number = 1;
  step: number = .01;
}

/**
 * Regression score, a single float.
 */
@registered
export class RegressionScore extends Scalar {
  parent?: string = undefined;
}

/**
 * Score of one or more target sequences.
 */
@registered
export class ReferenceScores extends _List {
  override default: number[] = [];

  /** Name of a TextSegment or ReferenceTexts field in the input. */
  parent?: string = undefined;
}

/**
 * Category or class label, a single string.
 */
@registered
export class CategoryLabel extends String {
  /** Optional vocabulary to specify allowed values.
   * If omitted, any value is accepted.
   */
  vocab?: string[] = undefined;  // Label names.
}
