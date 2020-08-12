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
 * Color palettes and related helpers, intended for use with d3.
 *
 * This is implemented as a TS library so that the same color list can be used
 * for different objects, such as div backgrounds or SVG text, with different
 * style names.
 */

interface ColorEntry {
  'color': string;
  'textColor': string;
}

type VizColorKey = 'pastel'|'bright'|'deep'|'dark';

/**
 * Colors for LIT.
 */
export class VizColor {
  // clang-format off
  protected static colors: {[key in VizColorKey]: ColorEntry[]} = {
    "pastel": [
      {color: '#ffd8c3', textColor: 'black'},
      {color: '#87cdf9', textColor: 'black'},
      {color: '#faf49f', textColor: 'black'},
      {color: '#cdc5ff', textColor: 'black'},
      {color: '#f6aea9', textColor: 'black'},
      {color: '#bdf4e7', textColor: 'black'},
      {color: '#edc4e6', textColor: 'black'},
      {color: '#dadce0', textColor: 'black'},  // "other"
    ],
    "bright": [
      {color: '#f0bd80', textColor: 'black'},
      {color: '#61aff7', textColor: 'black'},
      {color: '#ffe839', textColor: 'black'},
      {color: '#9b86ef', textColor: 'black'},
      {color: '#ff777b', textColor: 'black'},
      {color: '#7ddad3', textColor: 'black'},
      {color: '#ef96cd', textColor: 'black'},
      {color: '#b5bcc3', textColor: 'black'},  // "other"
    ],
    "deep": [
      {color: '#ff9230', textColor: 'black'},
      {color: '#3c7dbf', textColor: 'white'},
      {color: '#ffc700', textColor: 'black'},
      {color: '#7647ea', textColor: 'white'},
      {color: '#fc4f61', textColor: 'white'},
      {color: '#1f978a', textColor: 'white'},
      {color: '#b22a72', textColor: 'white'},
      {color: '#879695', textColor: 'white'},  // "other"
    ],
    "dark": [
      {color: '#ba4a0d', textColor: 'white'},
      {color: '#184889', textColor: 'white'},
      {color: '#b48014', textColor: 'white'},
      {color: '#270086', textColor: 'white'},
      {color: '#b12d33', textColor: 'white'},
      {color: '#006067', textColor: 'white'},
      {color: '#632440', textColor: 'white'},
      {color: '#515050', textColor: 'white'},  // "other"
    ],
  };
  // clang-format on

  /**
   * Get a color, keyed on version and index.
   */
  static getColor(version: VizColorKey, index: number|null): ColorEntry {
    const palette = VizColor.colors[version];
    if (index === null) {
      // last entry is 'other'
      return palette[palette.length - 1];
    } else {
      // return one of the normal entries
      return palette[index % (palette.length - 1)];
    }
  }
}
