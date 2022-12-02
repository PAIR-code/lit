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

// Some class properties are in snake_case to match their Python counterparts.
// We use _ prefixes to denote private LitTypes, consistent with
// their Python counterparts.
// tslint:disable:no-new-decorators class-name enforce-name-casing

import {AnnotationCluster, EdgeLabel, FeatureSalience as FeatureSalienceDType, ScoredTextCandidates, SequenceSalienceMap, SpanLabel, TokenSalience as TokenSalienceDType} from './dtypes';

/**
 * A dictionary of registered LitType names mapped to their constructor.
 * LitTypes are added using the @registered decorator.
 */
export const LIT_TYPES_REGISTRY: {[litType: string]: new () => LitType} = {};
function registered(target: new () => LitType) {
  LIT_TYPES_REGISTRY[target.name] = target;
}

const registryKeys: string[] = Object.keys(LIT_TYPES_REGISTRY);
/**
 * The types of all LitTypes in the registry, e.g.
 * 'StringLitType' | 'TextSegment' ...
 */
export type LitName = typeof registryKeys[number];

/** A type alias for the LitType class. */
export type LitClass = 'LitType';

/** A list of types of LitTypes. */
export type LitTypeTypesList = Array<typeof LitType>;

/**
 * Data classes used in configuring front-end components to describe
 * input data and model outputs.
 */
@registered
export class LitType {
  get name(): string {
    return this.constructor.name;
  }

  required: boolean = true;
  default: unknown = undefined;
  // If this type is created from an Annotator.
  annotated: boolean = false;
  show_in_data_table: boolean = false;
}

/** A type alias for LitType with an align property. */
export type LitTypeWithAlign = LitType&{align: string};

/** A type alias for LitType with a null idx property. */
export type LitTypeWithNullIdx = LitType&{null_idx: number};

/** A type alias for LitType with a parent property. */
export type LitTypeWithParent = LitType&{parent: string};

/** A type alias for LitType with a vocab property. */
export type LitTypeWithVocab = LitType&{vocab: string[]};

/**
 * A string LitType.
 */
@registered
export class StringLitType extends LitType {
  override default: string = '';
}

/**
 * Text input (untokenized), a single string.
 */
@registered
export class TextSegment extends StringLitType {
}

/**
 * An image, an encoded base64 ascii string (starts with 'data:image...').
 */
@registered
export class ImageBytes extends LitType {
}

/**
 * A JPEG image, as an encoded base64 ascii string
 * (starts with 'data:image/jpg...').
 */
@registered
export class JPEGBytes extends ImageBytes {
}

/**
 * A PNG image, as an encoded base64 ascii string
 * (starts with 'data:image/png...').
 */
@registered
export class PNGBytes extends ImageBytes {
}


/**
 * Generated (untokenized) text.
 */
@registered
export class GeneratedText extends TextSegment {
  parent?: string = undefined;
}

/**
 * A list type. Named `ListListType` to avoid conflicts with TypeScript List.
 */
@registered
export class ListLitType extends LitType {
  override default: unknown[] = [];
}

/**
 * A list of (text, score) tuples.
 */
class _StringCandidateList extends ListLitType {
  override default: ScoredTextCandidates = [];
}

/**
 * Multiple candidates for GeneratedText.
 */
@registered
export class GeneratedTextCandidates extends _StringCandidateList {
  /** Name of a TextSegment field to evaluate against. */
  parent?: string = undefined;
}

/**
 * Multiple candidates for TextSegment.
 */
@registered
export class ReferenceTexts extends _StringCandidateList {
}

/**
 * Multiple tokens with weight.
 */
@registered
export class TopTokens extends _StringCandidateList {
}


/**
 * TextSegment that should be interpreted as a URL.
 */
@registered
export class URLLitType extends TextSegment {
}

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
export class SearchQuery extends TextSegment {
}

/**
 * A list of strings.
 */
@registered
export class StringList extends ListLitType {
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
export class TokenTopKPreds extends ListLitType {
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
export class ReferenceScores extends ListLitType {
  override default: number[] = [];

  /** Name of a TextSegment or ReferenceTexts field in the input. */
  parent?: string = undefined;
}

/**
 * Category or class label, a single string.
 */
@registered
export class CategoryLabel extends StringLitType {
  /**
   * Optional vocabulary to specify allowed values.
   * If omitted, any value is accepted.
   */
  vocab?: string[] = undefined;  // Label names.
}

/**
 * A tensor type.
 */
class _Tensor extends LitType {
  override default: number[] = [];
}


/**
 * Multiclass predicted probabilities, as <float>[num_labels].
 */
@registered
export class MulticlassPreds extends _Tensor {
  /**
   * Vocabulary is required here for decoding model output.
   * Usually this will match the vocabulary in the corresponding label field.
   */
  vocab: string[] = [];
  /** Vocab index of negative (null) label. */
  null_idx?: number = undefined;
  /** CategoryLabel field in input. */
  parent?: string = undefined;
  /** Enable automatic sorting. */
  autosort?: boolean = false;
  /** Binary threshold, used to compute margin. */
  threshold?: number = undefined;

  get num_labels() {
    return this.vocab.length;
  }
}


/**
 * Sequence tags, aligned to tokens.
 * The data should be a list of string labels, one for each token.
 */
@registered
export class SequenceTags extends StringList {
  /** Name of Tokens field. **/
  align?: string = undefined;
}

/**
 * Span labels aligned to tokens.
 * Span labels can cover more than one token, may not cover all tokens in the
 * sentence, and may overlap with each other.
 */
@registered
export class SpanLabels extends ListLitType {
  /** Name of Tokens field. **/
  override default: SpanLabel[] = [];

  align: string = '';
  parent?: string = undefined;
}

/**
 * Edge labels between pairs of spans.
 *
 * This is a general form for structured prediction output; each entry consists
 * of (span1, span2, label).
 */
@registered
export class EdgeLabels extends ListLitType {
  override default: EdgeLabel[] = [];

  /** Name of Tokens field. **/
  align: string = '';
}

/**
 * Very general type for in-line text annotations.
 *
 * This is a more general version of SpanLabel, EdgeLabel, and other annotation
 * types, designed to represent annotations that may span multiple segments.
 *
 * The basic unit is dtypes.AnnotationCluster, which contains a label, optional
 *  score, and one or more SpanLabel annotations, each of which points to a
 * specific segment from the input.
 */
@registered
export class MultiSegmentAnnotations extends ListLitType {
  override default: AnnotationCluster[] = [];

  /** If true, treat as candidate list. */
  exclusive: boolean = false;
  /** If true, don't emphasize in visualization. */
  background: boolean = false;
}

/**
 * Embeddings or model activations, as fixed-length <float>[emb_dim].
 */
@registered
export class Embeddings extends _Tensor {
}

/**
 * Shared gradient attributes.
 */
class _GradientsBase extends _Tensor {
  /** Name of a Tokens field. */
  align?: string = undefined;
  /** Name of Embeddings field. */
  grad_for?: string = undefined;
  /**
   * Name of the field in the input that can be used to specify the target
   * class for the gradients.
   */
  grad_target_field_key?: string = undefined;
}

/**
 * 1D gradients with respect to embeddings.
 */
@registered
export class Gradients extends _GradientsBase {
}

/**
 * A single vector of <float>[enc_dim].
 */
class _InfluenceEncodings extends _Tensor {
  /** Class for computing gradients (string). */
  grad_target?: string = undefined;
}

/**
 * Per-token embeddings, as <float>[num_tokens, emb_dim].
 */
@registered
export class TokenEmbeddings extends _Tensor {
  /** Name of a Tokens field. */
  align?: string = undefined;
}

/**
 * Gradients with respect to per-token inputs, as <float>[num_tokens, emb_dim].
 */
@registered
export class TokenGradients extends _GradientsBase {
}

/**
 * Gradients with respect to per-pixel inputs, as a multidimensional array.
 */
@registered
export class ImageGradients extends _GradientsBase {
}

/**
 * One or more attention heads, as <float>[num_heads, num_tokens, num_tokens].
 */
@registered
export class AttentionHeads extends _Tensor {
  // Input and output Tokens fields; for self-attention these can
  // be the same.
  align_in: string = '';
  align_out: string = '';
}

/**
 * Offsets to align input tokens to wordpieces or characters.
 * offsets[i] should be the index of the first wordpiece for input token i.
 */
@registered
export class SubwordOffsets extends ListLitType {
  override default: number[] = [];
  /** Name of field in data spec. */
  align_in: string = '';
  /** Name of field in model output spec. */
  align_out: string = '';
}


/**
 * Sparse multi-label represented as a list of strings.
 */
@registered
export class SparseMultilabel extends StringList {
  /** Label names. */
  vocab?: string[] = undefined;
  /** Separator used for display purposes. */
  separator: string = ',';
}


/**
 * Sparse multi-label predictions represented as a list of tuples.
 * The tuples are of the label and the score.
 */
@registered
export class SparseMultilabelPreds extends _StringCandidateList {
  override default: ScoredTextCandidates = [];
  /** Label names. */
  vocab?: string[] = undefined;
  parent?: string = undefined;
}


/**
 * For matching spec fields.
 *
 * The front-end will perform spec matching and fill in the vocab field
 * accordingly. UI will materialize this to a dropdown-list.
 */
@registered
export class FieldMatcher extends LitType {
  /** Which spec to check, 'dataset', 'input', or 'output'. */
  spec: string = 'dataset';
  /** Types of LitType to match in the spec. */
  types: string|string[] = '';
  /** Names matched from the spec. */
  vocab?: string[] = undefined;
}


/**
 * For matching a single spec field.
 * UI will materialize this to a dropdown-list.
 */
@registered
export class SingleFieldMatcher extends FieldMatcher {
  override default: string = '';
}

/**
 * For matching multiple spec fields.
 * UI will materialize this to multiple checkboxes. Use this when the user needs
 * to pick more than one field in UI.
 */
@registered
export class MultiFieldMatcher extends FieldMatcher {
  /** Default names of selected items. */
  override default: string[] = [];
  /** Select all by default (overrides default). */
  select_all: boolean = false;
}

/**
 * Metadata about a returned salience map.
 */
@registered
export class Salience extends LitType {
  /** If the saliency technique is automatically run. */
  autorun: boolean = false;
  /** If the returned values are signed. */
  signed: boolean = false;
}

/**
 * Metadata about a returned token salience map.
 */
@registered
export class TokenSalience extends Salience {
  override default: TokenSalienceDType|undefined = undefined;
}

/**
 * Metadata about a returned feature salience map.
 */
@registered
export class FeatureSalience extends Salience {
  override default: FeatureSalienceDType|undefined = undefined;
}

/**
 * Metadata about a returned image saliency.
 * The data is returned as an image in the base64 URL encoded format, e.g.,
 * data:image/jpg;base64,w4J3k1Bfa...
 */
@registered
export class ImageSalience extends Salience {
}

/**
 * Metadata about a returned sequence salience map.
 */
@registered
export class SequenceSalience extends Salience {
  override default: SequenceSalienceMap|undefined = undefined;
}

/**
 * A boolean value.
 */
@registered
export class BooleanLitType extends LitType {
  override default: boolean = false;
}

/**
 * Represents data points of a curve.
 *
 * A list of tuples where the first and second elements of the tuple are the
 * x and y coordinates of the corresponding curve point respectively.
 */
@registered
export class CurveDataPoints extends LitType {
}

/**
 * Represents influential examples from the training set.
 *
 * This is as returned by a training-data attribution method like TracIn or
 * influence functions.
 *
 * This describes a generator component; values are
 * Sequence[Sequence[JsonDict]].
 */
@registered
export class InfluentialExamples extends LitType {
}

/** The method to use to determine the best value for a Metric. */
export enum MetricBestValue {
  HIGHEST = "highest",
  LOWEST = "lowest",
  NONE = "none",
  ZERO = "zero",
}

/** Score returned from the computation of a Metric. */
@registered
export class MetricResult extends LitType {
  override default: number = 0;
  description: string = '';
  best_value: MetricBestValue = MetricBestValue.NONE;
}
