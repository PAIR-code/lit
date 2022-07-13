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

// tslint:disable:enforce-comments-on-exported-symbols enforce-name-casing
import * as d3 from 'd3';
import {TemplateResult} from 'lit';

import {chunkWords, isLitSubtype} from './utils';

// tslint:disable-next-line:no-any
export type D3Selection = d3.Selection<any, any, any, any>;

export type LitClass = 'LitType';
export type LitName = 'type'|'LitType'|'String'|'TextSegment'|'GeneratedText'|
    'GeneratedTextCandidates'|'ReferenceTexts'|'URL'|'SearchQuery'|'Tokens'|
    'TokenTopKPreds'|'Scalar'|'RegressionScore'|'CategoryLabel'|
    'MulticlassPreds'|'SequenceTags'|'SpanLabels'|'EdgeLabels'|
    'MultiSegmentAnnotations'|'Embeddings'|'TokenGradients'|'TokenEmbeddings'|
    'AttentionHeads'|'SparseMultilabel'|'FieldMatcher'|'MultiFieldMatcher'|
    'Gradients'|'Boolean'|'TokenSalience'|'ImageBytes'|'SparseMultilabelPreds'|
    'ImageGradients'|'ImageSalience'|'SequenceSalience'|'ReferenceScores'|
    'FeatureSalience'|'TopTokens'|'CurveDataPoints'|'InfluentialExamples'|
    'GeneratedURL';

export const listFieldTypes: LitName[] =
    ['Tokens', 'SequenceTags', 'SpanLabels', 'EdgeLabels', 'SparseMultilabel'];

export interface LitType {
  __class__: LitClass|'type';
  __name__: LitName;
  __mro__: string[];
  parent?: string;
  align?: string;
  align_in?: string;
  align_out?: string;
  vocab?: string[];
  null_idx?: number;
  required?: boolean;
  annotated?: boolean;
  default?: string|string[]|number|number[];
  spec?: string;
  types?: LitName|LitName[];
  min_val?: number;
  max_val?: number;
  step?: number;
  exclusive?: boolean;
  background?: boolean;
  separator?: string;
  autorun?: boolean;
  signed?: boolean;
  mask_token?: string;
  token_prefix?: string;
  select_all?: boolean;
  autosort?: boolean;
  show_in_data_table?: boolean;
}

export interface Spec {
  [key: string]: LitType;
}

export interface ComponentInfo {
  configSpec: Spec;
  metaSpec: Spec;
  description?: string;
}

export interface DatasetInfo {
  size: number;
  spec: Spec;
  description?: string;
}

export interface ComponentInfoMap {
  [name: string]: ComponentInfo;
}

export interface DatasetInfoMap {
  [name: string]: DatasetInfo;
}

export interface CallConfig {
  //tslint:disable-next-line:no-any
  [option: string]: any;
}

export interface ModelSpec {
  input: Spec;
  output: Spec;
}

export interface ModelInfo {
  datasets: string[];
  generators: string[];
  interpreters: string[];
  spec: ModelSpec;
  description?: string;
}

export interface ModelInfoMap {
  [modelName: string]: ModelInfo;
}

export interface LitMetadata {
  models: ModelInfoMap;
  datasets: DatasetInfoMap;
  generators: ComponentInfoMap;
  interpreters: ComponentInfoMap;
  layouts: LitComponentLayouts;
  littypes: Spec;
  demoMode: boolean;
  defaultLayout: string;
  canonicalURL?: string;
  pageTitle?: string;
  inlineDoc?: string;
  onboardStartDoc?: string;
  onboardEndDoc?: string;
}

export interface Input {
  // tslint:disable-next-line:no-any
  [key: string]: any;
}

export interface IndexedInput {
  id: string;
  data: Input;
  meta: {source?: string; added?: boolean; parentId?: string;};
}

/**
 * Examples faceted by a given set of features.
 */
export interface FacetedData {
  data: IndexedInput[];
  /** Name to display */
  displayName?: string;
  /** What values were used as filters to get this data */
  facets?: FacetMap;
}

/**
 * Dictionary of multiple sets of faceted data.
 */
export interface GroupedExamples {
  [key: string]: FacetedData;
}


export interface Preds {
  // tslint:disable-next-line:no-any
  [key: string]: any;
}

export interface NumericResults {
  [key: string]: number;
}

export interface TopKResult {
  // tslint:disable-next-line:enforce-name-casing
  0: string;
  1: number;
}

export interface SpanLabel {
  start: number;  // inclusive
  end: number;    // exclusive
  label: string;
  align?: string;
}
export function formatSpanLabel(s: SpanLabel): string {
  // Add non-breaking control chars to keep this on one line
  // TODO(lit-dev): get it to stop breaking between ) and :; \u2060 doesn't work
  let formatted = `[${s.start}, ${s.end})`;
  if (s.align) {
    formatted = `${s.align} ${formatted}`;
  }
  if (s.label) {
    formatted = `${formatted}\u2060: ${s.label}`;
  }
  return formatted.replace(/\ /g, '\u00a0' /* &nbsp; */);
}

/**
 * Represents a directed edge between two mentions.
 * If span2 is null, interpret as a single span label.
 * See https://arxiv.org/abs/1905.06316 for more on this formalism.
 */
export interface EdgeLabel {
  span1: [number, number];   // inclusive, exclusive
  span2?: [number, number];  // inclusive, exclusive
  label: string|number;
}
export function formatEdgeLabel(e: EdgeLabel): string {
  const formatSpan = (s: [number, number]) => `[${s[0]}, ${s[1]})`;
  const span1Text = formatSpan(e.span1);
  const span2Text = e.span2 ? ' ← ' + formatSpan(e.span2) : '';
  // Add non-breaking control chars to keep this on one line
  // TODO(lit-dev): get it to stop breaking between ) and :; \u2060 doesn't work
  return `${span1Text}${span2Text}\u2060: ${e.label}`.replace(
      /\ /g, '\u00a0' /* &nbsp; */);
}

/**
 * Represents an annotation and its score, and the segment(s) is spans.
 */
export interface AnnotationCluster {
  label: string;
  score?: number;
  spans: SpanLabel[];
}
/** Formats an AnnotationCluster for textual display, e.g., in the DataTable. */
export function formatAnnotationCluster(ac: AnnotationCluster): string {
  return `${ac.label}${ac.score != null ? ` (${ac.score})` : ''}`;
}

/**
 * Element of GeneratedTextCandidates and ReferenceTexts fields.
 */
export type GeneratedTextCandidate = [string, number | null];

/**
 * Info about individual classifications including computed properties.
 */
export interface ClassificationResult {
  scores: number[];
  predicted_class: string;
  correct?: boolean;
}

export interface ClassificationResults {
  [key: string]: ClassificationResult;
}

/**
 * Info about individual regressions including computed properties.
 */
export interface RegressionResult {
  score: number;
  error?: number;
  squared_error?: number;
}

export interface RegressionResults {
  [key: string]: RegressionResult;
}

/**
 * Type for d3 scale object used for datapoint coloring.
 */
export interface D3Scale {
  (value: number|string): string;
  domain: () => Array<number|string>;
}

/**
 * Object to specify a coloring option for datapoints.
 */
export interface ColorOption {
  name: string;
  // Function to convert a datapoint into a number or string to be provided to
  // the scale property for conversion to a color.
  getValue: (input: IndexedInput) => number | string;
  // D3 scale object to convert a value returned from getValue into a color
  // string.
  scale: D3Scale;
}

export interface NumericSetting {
  [key: string]: number;
}

/**
 * Generic wrapper type for constructors, used in the DI system
 */
// tslint:disable:no-any
export type Constructor<T> = {
  new (...args: any[]): T
}|((...args: any[]) => T)|Function;
// tslint:enable:no-any

/**
 * Information on a facet for grouping examples. The value is either a string
 * or numeric value to match, or a bucket of a numerical range, [min, max).
 * The displayVal is used for displaying the facet information.
 */
export interface FacetInfo {
  val: string|number|number[];
  displayVal: string;
}

/**
 * Dictionary of features (e.g., features of an Input). Used for grouping sets
 * of Inputs in the slices module, metrics module, etc. Features may be strings
 * (e.g. 'neutral', 'entailment'), or numerical features (e.g., sentence
 * similarity in stsb), or a range of numbers used for bucketed numbers.
 */
export interface FacetMap {
  [fieldName: string]: FacetInfo;
}

/**
 * Identity of a module triggering an action in this service.
 * Usually, set to 'this' from the calling module, so it can distinguish
 * selection updates from itself vs another module.
 */
export type ServiceUser = object;

/**
 * We can't define abstract static properties/methods in typescript, so we
 * define an interface to emulate the LitModuleType.
 */
export interface LitModuleClass {
  title: string;
  template:
      (modelName?: string, selectionServiceIndex?: number) => TemplateResult;
  shouldDisplayModule: (modelSpecs: ModelInfoMap, datasetSpec: Spec) => boolean;
  duplicateForExampleComparison: boolean;
  duplicateForModelComparison: boolean;
  duplicateAsRow: boolean;
  numCols: number;
  collapseByDefault: boolean;
}

/**
 * Get the default value for a spec field.
 */
export function defaultValueByField(key: string, spec: Spec) {
  const fieldSpec: LitType = spec[key];
  // Explicitly check against undefined, as this value is often false-y if set.
  if (fieldSpec.default !== undefined) {
    return fieldSpec.default;
  }
  // TODO(lit-dev): remove these and always use the spec default value.
  if (isLitSubtype(fieldSpec, 'Scalar')) {
    return 0;
  }

  if (isLitSubtype(fieldSpec, listFieldTypes)) {
    return [];
  }

  if (isLitSubtype(fieldSpec, 'ImageBytes')) {
    return '';
  }

  const stringFieldTypes: LitName[] = ['TextSegment', 'CategoryLabel'];
  if (isLitSubtype(fieldSpec, stringFieldTypes)) {
    return '';
  }
  console.log(
      'Warning: default value requested for unrecognized input field type', key,
      fieldSpec);
  return '';
}

/**
 * Dictionary of lit layouts. See LitComponentLayout
 */
export declare interface LitComponentLayouts {
  [key: string]: LitComponentLayout|LitCanonicalLayout;
}

// LINT.IfChange
/**
 * Module with additional config options.
 */
export declare interface LitModuleConfig {
  module: (keyof HTMLElementTagNameMap);
  requiredForTab?: boolean;
  // TODO(b/172979677): support title, duplicateAsRow, numCols,
  // and startMinimized.
}

/**
 * As above, but guaranteeing fields have been populated.
 */
export declare interface ResolvedModuleConfig {
  module: (keyof HTMLElementTagNameMap);
  constructor: LitModuleClass;
  requiredForTab: boolean;
  title: string;
  // TODO(b/172979677): support title, duplicateAsRow, numCols,
  // and startMinimized.
}

/**
 * Leaf values in a LitComponentLayout or LitCanonicalLayout.
 */
export type LitComponentSpecifier =
    (keyof HTMLElementTagNameMap)|LitModuleConfig;


export declare interface LitTabGroupLayout {
  [tabName: string]: LitComponentSpecifier[];
}

/**
 * A layout is defined by a set of modules arranged into tabs and groups.
 *
 * This is a legacy layout format, where components contains several keys
 * including 'Main', and values are lists of module classes.
 * - The 'Main' group will appear in the upper half of the UI
 * - The remaining groups will appear in the lower half of the UI,
 *   with a tab bar to select the active group.
 *
 * See layout.ts for examples.
 */
export declare interface LitComponentLayout {
  components: LitTabGroupLayout;
  layoutSettings?: LayoutSettings;
  description?: string;
}

/**
 * UI layout in canonical form.
 *
 * This has explicit tab groups for the upper and lower UI areas.
 * Recommended for new layouts.
 *
 * See layout.ts for examples.
 */
export declare interface LitCanonicalLayout {
  upper: LitTabGroupLayout;
  lower: LitTabGroupLayout;
  layoutSettings: LayoutSettings;
  description: string;
}

/**
 * Miscellaneous render settings (e.g., whether to render a toolbar)
 * for a given layout.
 */
export declare interface LayoutSettings {
  hideToolbar?: boolean;
  mainHeight?: number;
  centerPage?: boolean;
}

/**
 * Convert a layout to canonical form.
 * TODO(lit-dev): deprecate this once we convert all client and demo layouts.
 * TODO(lit-dev): move this to Python.
 */
export function canonicalizeLayout(layout: LitComponentLayout|
                                   LitCanonicalLayout): LitCanonicalLayout {
  if (!layout.hasOwnProperty('components')) {
    return layout as LitCanonicalLayout;
  }
  // Legacy layout to convert.
  layout = layout as LitComponentLayout;

  const canonicalLayout: LitCanonicalLayout = {
    upper: {},
    lower: {},
    layoutSettings: layout.layoutSettings ?? {},
    description: layout.description ?? '',
  };

  // Handle upper layout.
  canonicalLayout.upper['Main'] = layout.components['Main'];

  // Handle lower layout.
  for (const tabName of Object.keys(layout.components)) {
    if (tabName === 'Main') continue;
    canonicalLayout.lower[tabName] = layout.components[tabName];
  }

  return canonicalLayout;
}

// LINT.ThenChange(../../api/layout.py)

/** Display name for the "no dataset" dataset in settings. */
export const NONE_DS_DISPLAY_NAME = 'none';
/** Key for the "no dataset" dataset in settings. */
export const NONE_DS_DICT_KEY = '_union_empty';

/**
 * Display name for dataset.
 */
export function datasetDisplayName(name: string): string {
  return name === NONE_DS_DICT_KEY ? NONE_DS_DISPLAY_NAME : name;
}

/**
 * CSS class name for module-internal divs that should have syncronized
 * scrolling between duplicated modules. See widget_group.ts and lit_module.ts
 * for its use.
 */
export const SCROLL_SYNC_CSS_CLASS = 'scroll-sync';

/**
 * Formats the following types for display in the data table:
 * string, number, boolean, string[], number[], (string|number)[]
 */
export function formatForDisplay(
    // tslint:disable-next-line:no-any
    input: any, fieldSpec?: LitType, limitWords?: boolean): string|number {
  if (input == null) return '';

  // Handle SpanLabels, if field spec given.
  // TODO(lit-dev): handle more fields this way.
  if (fieldSpec != null && isLitSubtype(fieldSpec, 'SpanLabels')) {
    const formattedTags = (input as SpanLabel[]).map(formatSpanLabel);
    return formattedTags.join(', ');
  }
  // Handle EdgeLabels, if field spec given.
  if (fieldSpec != null && isLitSubtype(fieldSpec, 'EdgeLabels')) {
    const formattedTags = (input as EdgeLabel[]).map(formatEdgeLabel);
    return formattedTags.join(', ');
  }
  // Handle MultiSegmentAnnotations, if field spec given.
  if (fieldSpec != null && isLitSubtype(fieldSpec, 'MultiSegmentAnnotations')) {
    const formattedTags =
        (input as AnnotationCluster[]).map(formatAnnotationCluster);
    return formattedTags.join(', ');
  }
  const formatNumber = (item: number) =>
      Number.isInteger(item) ? item : Number(item.toFixed(4));

  // Generic data, based on type of input.
  if (Array.isArray(input)) {
    const strings = input.map((item) => {
      if (typeof item === 'number') {
        return formatNumber(item);
      }
      if (limitWords) {
        return chunkWords(item);
      }
      return `${item}`;
    });
    return `${strings.join(', ')}`;
  }

  if (typeof input === 'boolean') {
    return formatBoolean(input);
  }

  if (typeof input === 'number') {
    return formatNumber(input);
  }

  // Fallback: just coerce to string.
  if (limitWords) {
    return chunkWords(input);
  }
  return `${input}`;
}

/**
 * Formats a boolean value for display.
 */
export function formatBoolean(val: boolean): string {
  return val ? '✔' : ' ';
}
