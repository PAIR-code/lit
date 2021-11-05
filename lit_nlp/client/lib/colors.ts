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

import * as d3 from 'd3';

interface ColorEntry {
  color: string;
  textColor: string;
}

type VizPaletteKey = 'pastel'|'bright'|'deep'|'dark';
type VizColorKey =
  'orange'|'blue'|'yellow'|'purple'|'coral'|'teal'|'magenta'|'other';
type LitBrandPaletteKey = 'cyea'|'mage'|'bric'|'neutral';
type LitTonalPaletteKey = 'primary'|'secondary'|'tertiary';
type LitMajorTonalPaletteKey = LitTonalPaletteKey|'neutral-variant';
type ColorValue =
  '50'|'100'|'200'|'300'|'400'|'500'|'600'|'700'|'800'|'900';

const FULL_COLOR_VALUES: ColorValue[] = [
  '50', '100', '200', '300', '400', '500', '600', '700', '800', '900'
];

const SMALL_COLOR_VALUES: ColorValue[] = [
  '50', '100', '200', '300', '400', '500', '600', '700', '800', '900'
];

const ERROR_COLOR_VALUES: ColorValue[] = [
  '50', '500', '600', '700'
];

const VIZ_COLOR_VALUES: VizColorKey[] = [
  'orange', 'blue', 'yellow', 'purple', 'coral', 'teal', 'magenta', 'other'
];

/**
 * Returns a ColorEntry at the index in th given palette.
 *
 * This function is a riff on the proposed Array.prototype.at() method. It
 * provides the bsame basic functionality in that it accepts positive and
 * negative numbers as an index (which it truncates prior to accessing to ensure
 * an integer value is used), but differs in that it takes the remainder of the
 * index after division with palette.length prior to accessing the value in the
 * palette. This ensures that values that overflow (are greater than
 * palette.length) or underflow (are less than 0) will still return a color,
 * which is useful for procedural access of colors in a palette, such as in
 * determining a color for use in data visualization.
 */
function at (palette:ColorEntry[], index:number): ColorEntry {
  index = Math.trunc(index);
  index = index % palette.length;
  if (index < 0) { index = index + palette.length; }
  return palette[index];
}

/**
 * Creates a D3-compatible color ramp function from a list of hex color values
 * using D3's interpolateRgbBasis() function.
 */
function ramp (range:string[]): (t:number) => string {
  return d3.interpolateRgbBasis(range);
}

/**
 * LIT Brand colors by palette with text color
 */
export const BRAND_COLORS: {[palette in LitBrandPaletteKey]: ColorEntry[]} = {
  cyea:[
    { color:"#EDFFFA", textColor:'black' },
    { color:"#CDF2FA", textColor:'black' },
    { color:"#AFE6DE", textColor:'black' },
    { color:"#7BCCCC", textColor:'black' },
    { color:"#52A6B3", textColor:'black' },
    { color:"#348199", textColor:'white' },
    { color:"#1F6180", textColor:'white' },
    { color:"#114566", textColor:'white' },
    { color:"#092F4D", textColor:'white' },
    { color:"#041D33", textColor:'white' },
  ],
  mage:[
    { color:"#FFF5F7", textColor:'black' },
    { color:"#FEEAEF", textColor:'black' },
    { color:"#FED5E0", textColor:'black' },
    { color:"#F9A9C0", textColor:'black' },
    { color:"#EF7CA1", textColor:'black' },
    { color:"#E25287", textColor:'white' },
    { color:"#CE2F75", textColor:'white' },
    { color:"#B7166A", textColor:'white' },
    { color:"#800060", textColor:'white' },
    { color:"#470046", textColor:'white' },
  ],
  bric:[
    { color:"#FDF8EA", textColor:'black' },
    { color:"#F9F1DC", textColor:'black' },
    { color:"#EFD9AB", textColor:'black' },
    { color:"#E4BC78", textColor:'black' },
    { color:"#D59441", textColor:'black' },
    { color:"#C26412", textColor:'white' },
    { color:"#A93D00", textColor:'white' },
    { color:"#8B2100", textColor:'white' },
    { color:"#6A0C00", textColor:'white' },
    { color:"#470000", textColor:'white' },
  ],
  neutral:[
    { color:"#F8F9FA", textColor:'black' },
    { color:"#F1F3F4", textColor:'black' },
    { color:"#E8EAED", textColor:'black' },
    { color:"#DADCE0", textColor:'black' },
    { color:"#BDC1C6", textColor:'black' },
    { color:"#9AA0A6", textColor:'white' },
    { color:"#80868B", textColor:'white' },
    { color:"#5F6368", textColor:'white' },
    { color:"#3C4043", textColor:'white' },
    { color:"#202124", textColor:'white' },
  ]
};

/**
 * Gets the ColorEntry for a color in LIT's Brand family
 */
export function getBrandColor (version:LitBrandPaletteKey,
                               id:number|ColorValue): ColorEntry {
  const palette = BRAND_COLORS[version];
  const index = typeof id === 'number' ? id : FULL_COLOR_VALUES.indexOf(id);
  return at(palette, index);
}

/**
 * List of color hex values from Brand Cyea palette
 */
export const CYEA_COLORS = BRAND_COLORS.cyea.map(c => c.color);

/**
 * Continuous color ramp from Cyea-200 thorugh Cyea-900
 */
export const CYEA_RAMP = ramp(CYEA_COLORS.slice(2));

/**
 * LIT Major Tonal colors by palette with text color
 */
export const MAJOR_TONAL_COLORS:
    {[palette in (LitTonalPaletteKey|'neutral-variant')]: ColorEntry[]} = {
  primary:[
    { color:"#EEFBF8", textColor:"black" },
    { color:"#D8F2ED", textColor:"black" },
    { color:"#C0E7E3", textColor:"black" },
    { color:"#A1D2D4", textColor:"black" },
    { color:"#72AEB9", textColor:"black" },
    { color:"#48879C", textColor:"white" },
    { color:"#326882", textColor:"white" },
    { color:"#284E67", textColor:"white" },
    { color:"#1D3649", textColor:"white" },
    { color:"#142838", textColor:"white" },
  ],
  secondary:[
    { color:"#FFF5F7", textColor:"black" },
    { color:"#FAEDF0", textColor:"black" },
    { color:"#F7DBE4", textColor:"black" },
    { color:"#EDBDCD", textColor:"black" },
    { color:"#E091AC", textColor:"black" },
    { color:"#CC6990", textColor:"white" },
    { color:"#BE4079", textColor:"white" },
    { color:"#9D2D69", textColor:"white" },
    { color:"#651A54", textColor:"white" },
    { color:"#3B0A3C", textColor:"white" },
  ],
  tertiary:[
    { color:"#FCF8EF", textColor:"black" },
    { color:"#F9F7ED", textColor:"black" },
    { color:"#EDDEBF", textColor:"black" },
    { color:"#E0C9A2", textColor:"black" },
    { color:"#CEA269", textColor:"black" },
    { color:"#B6763E", textColor:"white" },
    { color:"#A14C1C", textColor:"white" },
    { color:"#7E351F", textColor:"white" },
    { color:"#58211B", textColor:"white" },
    { color:"#3B0A0B", textColor:"white" },
  ],
  'neutral-variant':[
    { color:"#F0F7F7", textColor:"black" },
    { color:"#EBF3F2", textColor:"black" },
    { color:"#E3ECED", textColor:"black" },
    { color:"#CCD9DD", textColor:"black" },
    { color:"#B5C6CA", textColor:"black" },
    { color:"#9CAFB4", textColor:"white" },
    { color:"#72868F", textColor:"white" },
    { color:"#596C75", textColor:"white" },
    { color:"#3F5259", textColor:"white" },
    { color:"#222C35", textColor:"white" },
  ]
};

/**
 * Gets the ColorEntry for a color in LIT's Major Tonal family
 */
export function getMajorTonalColor (version:LitMajorTonalPaletteKey,
                                    id:number|ColorValue): ColorEntry {
  const palette = MAJOR_TONAL_COLORS[version];
  const index = typeof id === 'number' ? id : FULL_COLOR_VALUES.indexOf(id);
  return at(palette, index);
}

/**
 * List of color hex values from Major Tonal Primary palette
 */
export const PRIMARY_COLORS = MAJOR_TONAL_COLORS.primary.map(p => p.color);

/**
 * Continuous color ramp from Major Tonal Primary-200 thorugh Primary-900
 */
export const PRIMARY_RAMP = ramp(PRIMARY_COLORS.slice(2));

/**
 * LIT Minor Tonal colors by palette with text color
 */
export const MINOR_TONAL_COLORS:
    {[palette in (LitTonalPaletteKey)]: ColorEntry[]} = {
  primary:[
    { color:"#F5F9FA", textColor:"black" },
    { color:"#EFF5F7", textColor:"black" },
    { color:"#E9F1F4", textColor:"black" },
    { color:"#E7F0F3", textColor:"black" },
    { color:"#E3EDF1", textColor:"black" }
  ],
  secondary:[
    { color:"#F9F2F7", textColor:"black" },
    { color:"#F5EBF2", textColor:"black" },
    { color:"#F1E3EE", textColor:"black" },
    { color:"#F0E0EC", textColor:"black" },
    { color:"#EDDBE9", textColor:"black" }
  ],
  tertiary:[
    { color:"#FAF4F3", textColor:"black" },
    { color:"#F6EDEB", textColor:"black" },
    { color:"#F2E7E3", textColor:"black" },
    { color:"#F3E4E1", textColor:"black" },
    { color:"#EFE0DB", textColor:"black" }
  ]
};

/**
 * Gets the ColorEntry for a color in LIT's Minor Tonal family
 */
export function getMinorTonalColor (version:LitTonalPaletteKey,
                                    id:number|ColorValue): ColorEntry {
  const palette = MINOR_TONAL_COLORS[version];
  const index = typeof id === 'number' ? id : SMALL_COLOR_VALUES.indexOf(id);
  return at(palette, index);
}

/**
 * Error colors derived from Google Red (50, 500, 600, 700)
 */
export const ERROR_COLORS: ColorEntry[] = [
  { color:"#FCE8E6", textColor:'black' },
  { color:"#EA4335", textColor:'white' },
  { color:"#D93025", textColor:'white' },
  { color:"#C5221F", textColor:'white' }
];

/**
 * Gets the ColorEntry for a color in LIT's Major Tonal family
 */
export function getErrorColor (id:number|ColorValue): ColorEntry {
  const index = typeof id === 'number' ? id : ERROR_COLOR_VALUES.indexOf(id);
  return at(ERROR_COLORS, index);
}

/**
 * Colors for LIT.
 */
export const VIZ_COLORS: {[key in VizPaletteKey]: ColorEntry[]} = {
  pastel: [
    {color: '#ffd8c3', textColor: 'black'},
    {color: '#87cdf9', textColor: 'black'},
    {color: '#faf49f', textColor: 'black'},
    {color: '#cdc5ff', textColor: 'black'},
    {color: '#f6aea9', textColor: 'black'},
    {color: '#bdf4e7', textColor: 'black'},
    {color: '#edc4e6', textColor: 'black'},
    {color: '#dadce0', textColor: 'black'},  // "other"
  ],
  bright: [
    {color: '#f0bd80', textColor: 'black'},
    {color: '#61aff7', textColor: 'black'},
    {color: '#ffe839', textColor: 'black'},
    {color: '#9b86ef', textColor: 'black'},
    {color: '#ff777b', textColor: 'black'},
    {color: '#7ddad3', textColor: 'black'},
    {color: '#ef96cd', textColor: 'black'},
    {color: '#b5bcc3', textColor: 'black'},  // "other"
  ],
  deep: [
    {color: '#ff9230', textColor: 'black'},
    {color: '#3c7dbf', textColor: 'white'},
    {color: '#ffc700', textColor: 'black'},
    {color: '#7647ea', textColor: 'white'},
    {color: '#fc4f61', textColor: 'white'},
    {color: '#1f978a', textColor: 'white'},
    {color: '#b22a72', textColor: 'white'},
    {color: '#879695', textColor: 'white'},  // "other"
  ],
  dark: [
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

/**
 * Gets the ColorEntry for a color in VizColor family.
 *
 * This function has special behavior to support common data visualization use
 * cases in LIT, specifically:
 *
 *   1. If the value of id is a positive number, it will be truncated and the
 *      the index will be determined by taking the reminder of the truncated
 *      integer divided by (palette.length - 1), which ensures the function
 *      returns one of the "normal" colors.
 *   2. Otherwise, we find the value for that id according to its order in the
 *      VisColor palette array. When the value of id is a VisColorKey, i.e., the
 *      name of that color in the palette, this funciton returns the specified
 *      color. However, if the value of id is a neagtive number, then the number
 *      will not be a vlaid name and indexOf() will return -1, which ensures
 *      that all negative numbers return the "other" color for that palette when
 *      the color is sccessed with at(). This special behavior is a convenient
 *      way to semanically distinguish between "normal" and "other" values in
 *      data visualization applications.
 */
export function getVizColor (version: VizPaletteKey,
                             id: number|VizColorKey): ColorEntry {
  const palette = VIZ_COLORS[version];
  if (typeof id === 'number' && id > -1) {
    // Return one of normal colors
    return palette[Math.trunc(id) % (palette.length - 1)];
  } else {
    const index = VIZ_COLOR_VALUES.indexOf(id as VizColorKey);
    return at(palette, index);
  }
}

/**
 * List of color hex values for normal colors from VizColors Pastel palette
 */
export const VIZ_COLORS_PASTEL = VIZ_COLORS.pastel
  .slice(0, -1).map(p => p.color);

/**
 * List of color hex values for normal colors from VizColors Bright palette
 */
export const VIZ_COLORS_BRIGHT = VIZ_COLORS.bright
  .slice(0, -1).map(b => b.color);

/**
 * List of color hex values for normal colors from VizColors Deep palette
 */
export const VIZ_COLORS_DEEP = VIZ_COLORS.deep
  .slice(0, -1).map(d => d.color);

/**
 * List of color hex values for normal colors from VizColors Dark palette
 */
export const VIZ_COLORS_DARK = VIZ_COLORS.dark
  .slice(0, -1).map(d => d.color);
