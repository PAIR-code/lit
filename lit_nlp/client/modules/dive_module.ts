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
 * @fileoverview Dive enables exploration of data across 2-dimensional facets.
 */

// tslint:disable:no-new-decorators
// taze: ResizeObserver from //third_party/javascript/typings/resize_observer_browser
import {css, html} from 'lit';
import {customElement, query} from 'lit/decorators';
import {Scene, Selection, SpriteView, TextSelection} from 'megaplot';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {colorToRGB, getBrandColor} from '../lib/colors';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {GroupedExamples, IndexedInput} from '../lib/types';
import {FacetingConfig, FacetingMethod} from '../services/group_service';
import {ColorService, DataService, FocusService, GroupService, SelectionService} from '../services/services';

const CELL_PADDING = 8;
const COLOR_TEXT = colorToRGB('black');
const COLOR_PRIMARY_BORDER = colorToRGB(getBrandColor('cyea', '700').color);
const COLOR_HOVER_FILL = colorToRGB(getBrandColor('mage', '400').color);
const COLOR_PINNED_BORDER = colorToRGB(getBrandColor('mage', '700').color);
const COLOR_NONE = [0, 0, 0, 0];
const HEURISTIC_SCALE_FACTOR = 0.7;
const TEXT_SIZE = 14;
const ZOOM_IN_FACTOR = 1.1;
const ZOOM_OUT_FACTOR = 0.9;
const ZOOM_WHEEL_FACTOR = -0.05;

/**
 * A map from the feature facet display name to the number of dots in a cell
 * (i.e., its size) in a dimension (columns = width, rows = height).
 *
 * Cell sizes are computed in `this.configFacets()` as either the square root of
 * `facet.data.length` for width, or `facet.data.length / sqrt` for height.
 */
interface FeatureSizes {
  [val: string]: {
    /** The size of this column or row. */
    size: number;
    /** The sum of the width or height of the preceeding columns or rows. */
    sumTo: number
  };
}

/**
 * A map from the feature selected for the rows and columns to the FeatureSizes
 * associated with that feature. Generated/used in `this.configFacets()`.
 */
interface GroupFeaturesSizes {
  columns: FeatureSizes;
  rows: FeatureSizes;
}

/** Position in Megaplot's 2D space. */
interface Position2D {
  x: number;
  y: number;
}

/** Internal representation of a datapoint to visualize in the matrix. */
interface Dot {
  id: string;
  /** Index of the cell in the matrix, used to compute pixel position. */
  pixel: Position2D;
  /** Position of the dot Megaplot world space. */
  world: Position2D;
}

/** Labels used to annotate columns and rows in the matrix. */
interface Label {
  text: string;
  pixel: Position2D;
  world: Position2D;
}

/** Megaplot Selections used to create the Dive visualization. */
interface DiveSelections {
  /** The column labels for the matrix. */
  columnLabels?: TextSelection<Label>;
  /** The row labels for the matrix. */
  rowLabels?: TextSelection<Label>;
  /** The datapoints in the dataset. */
  points?: Selection<Dot>;
}

function bindLabel(sprite: SpriteView, label: Label) {
  sprite.FillColor = COLOR_TEXT;
  sprite.SizePixel = TEXT_SIZE;
  sprite.PositionWorld = label.world;
  sprite.PositionPixel = label.pixel;
}

function keySorter(a: string, b: string): number {
  return a.localeCompare(b);
}

/** Default exit handler for sprites in the visualization. */
function spriteExitHandler (sprite: SpriteView) {
  sprite.FillColor = COLOR_NONE;
  sprite.SizePixel = 0;
}

/**
 * A LIT module that visualizes datapoints clusters based on their properties.
 */
@customElement('dive-module')
export class DiveModule extends LitModule {
  static override title = 'Dive';
  static override numCols = 3;
  static override shouldDisplayModule = () => true;

  static override get styles() {
    const styles = css`
      .scene {
        height: 100%;
        width: 100%;
        cursor: crosshair;
      }

      select.limit-width {
        width: 150px;
        max-width: 150px;
      }

      .dive-controls {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
      }

      .dive-controls .feature-selectors {
        display: flex;
        flex-direction: row;
      }

      .dive-controls .icon-button {
        font-size: 16px;
      }
    `;

    return [sharedStyles, styles];
  }

  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`<dive-module model=${model} .shouldReact=${shouldReact}
                selectionServiceIndex=${selectionServiceIndex}>
              </dive-module>`;

  private readonly colorService = app.getService(ColorService);
  private readonly dataService = app.getService(DataService);
  private readonly focusService = app.getService(FocusService);
  private readonly groupService = app.getService(GroupService);
  private readonly pinnedSelectionService =
      app.getService(SelectionService, 'pinned');

  /**
   * An index into `this.groupService.denseFeatureNames`, used to find the
   * feature to bin the space by along the X axis.
   */
  @observable private colsFeatureIdx = 1;

  /**
   * An index into `this.groupService.denseFeatureNames`, used to find the
   * feature to bin the space by along the Y axis.
   */
  @observable private rowsFeatureIdx = 0;


  /** The points to visualize in the matrix. */
  private points = new Map<string, Dot>();
  /** The dots to bind in the matrix. */
  private dots: Dot[] = [];
  /** The column labels for the matrix. */
  private columns: Label[] = [];
  /** The row labels for the matrix. */
  private rows: Label[] = [];

  private readonly colorValueMap = new Map<string, number|string>();
  private readonly resizeObserver = new ResizeObserver(() => {this.resize();});

  /** The `<div>` into which the Megaplot scene is drawn. */
  @query('div.scene') private readonly container?: HTMLDivElement;

  /** The Megaplot Scene into which the visualization is rendered. */
  private scene?: Scene;
  private mouseDown?: Position2D;

  /** The Megaplot Selections that get rendered into the visualziation. */
  private readonly selections: DiveSelections = Object.seal({
    columnLabels: undefined,
    rowLabels: undefined,
    points: undefined
  });

  /**
   * Configures the Megaplot Selections for each facet in `this.groupedExamples`
   * using the grouped facets method, which produces a 2D matrix of dot groups,
   * similar to small multiples visualizations.
   *
   * The GroupService returns groups of examples at the intersections of facet
   * parameters. The examples in each group are displayed as a rectangle of dots
   * in a cell in the matrix.
   *
   * The dot group's width is equivalent to the ceiling of the square root of
   * the number of examples, and the height is equivalent to the ceiling of the
   * number of examples divided by the width.
   *
   * The size of the cell is determined by the width and height of the largest
   * dot group in its row and column. As such, the dot group needs to be
   * centered inside the cell by computing the delta between the dot group's
   * size and the cell's size.
   */
  private configure() {
    const colFeature = this.groupService.denseFeatureNames[this.colsFeatureIdx];
    const rowFeature = this.groupService.denseFeatureNames[this.rowsFeatureIdx];
    if (colFeature == null || rowFeature == null) return;

    const bindDot = (sprite: SpriteView, dot: Dot) => {
      const isHovered =
          this.focusService.focusData?.datapointId === dot.id;
      const isPinned =
          this.pinnedSelectionService.primarySelectedId === dot.id;
      const isPrimary =
          this.selectionService.primarySelectedId === dot.id;
      const isSelected =
          this.selectionService.selectedIds.includes(dot.id) &&
          !(isHovered || isPinned || isPrimary);
      const isSpecial = isHovered || isPinned || isPrimary || isSelected;
      const noSelections = this.selectionService.selectedIds.length === 0;

      const input = this.appState.getCurrentInputDataById(dot.id);
      const colorString = this.colorService.getDatapointColor(input);
      const color = colorToRGB(colorString);

      sprite.BorderColor = isPinned  ? COLOR_PINNED_BORDER :
                           isPrimary ? COLOR_PRIMARY_BORDER : color;
      sprite.BorderColorOpacity = isSpecial ? 1 : 0.25;
      sprite.BorderRadiusPixel = (isPinned || isPrimary) ? 2 : 0;
      sprite.BorderPlacement = 1;
      sprite.FillColor = (isHovered || isPinned) ? COLOR_HOVER_FILL : color;
      sprite.FillColorOpacity = (noSelections || isSpecial) ? 1 : 0.25;
      sprite.PositionWorld = dot.world;
      sprite.PositionPixel = dot.pixel;
      sprite.Sides = 2;
      sprite.SizeWorld = stepSize * .9;
      sprite.SizePixel = 0;
    };

    // TODO(b/202181874): Explore alternative sorting based on bucketing.
    const sortInputs = (a: IndexedInput, b: IndexedInput): number => {
      const aVal = this.colorValueMap.get(a.id)!;
      const bVal = this.colorValueMap.get(b.id)!;
      if (aVal > bVal) {
        return 1;
      } else if (aVal < bVal) {
        return -1;
      } else {
        return 0;
      }
    };

    const groupedExamples = this.groupExamples([rowFeature, colFeature]);
    const sizes = this.matrixSizes(groupedExamples, colFeature, rowFeature);
    const colSizes = sizes.columns;
    const colFeatureVals = Object.keys(colSizes).sort(keySorter);
    const rowSizes = sizes.rows;
    const rowFeatureVals = Object.keys(rowSizes).sort(keySorter);
    /**
     * The step size used to position dots in the grid.
     *
     * Dots in each grid are positioned with the same spacing, in Megaplot world
     * coorinates. Since the initial domain of Megaplot's world coordinates is a
     * unit square (i.e., width/height have a domain of [-0.5, 0.5]) the step
     * size is computed as 1 over sum of the column feature facet sizes, since
     * the width (square root) will always be greater than or equal to the
     * height (examples.length / square root) of the dot grid.
     */
    const stepSize = 1 / Object.values(colSizes).reduce(
        (acc, column) => acc + column.size, 0);

    this.columns = colFeatureVals.map((text, index) => {
      const colDimension = colSizes[text];
      const colSumToValue = colDimension.sumTo;
      const midpoint = colDimension.size / 2;
      const worldX = (colSumToValue + midpoint) * stepSize - 0.5;
      const pixelX = CELL_PADDING * colFeatureVals.indexOf(text);
      return {
        text,
        pixel: {x: pixelX, y: 0},
        world: {x: worldX, y: 0.5}
      } as Label;
    });

    this.rows = rowFeatureVals.map((text, index) => {
      const rowDimension = rowSizes[text];
      const rowSumToValue = rowDimension.sumTo;
      const midpoint = rowDimension.size / 2;
      const worldX = -(0.5 + stepSize);
      const worldY = 0.5 - (rowSumToValue + midpoint) * stepSize;
      const pixelY = CELL_PADDING * rowFeatureVals.indexOf(text);
      return {
        text,
        pixel: {x: 0, y: pixelY},
        world: {x: worldX, y: worldY}
      } as Label;
    });

    for (const [name, group] of Object.entries(groupedExamples)) {
      // TODO(b/251452293): Probably don't need this once GroupService is fixed.
      if (!(colFeature in group.facets && rowFeature in group.facets)) {
        console.warn(
          `Skipping "${name}". Missing ${colFeature} or ${rowFeature}. Has ${
            Object.keys(group.facets).join(' ')
          }. Contains ${group.data.length} element(s).`, group);
        continue;
      }

      const sqrt = Math.ceil(Math.sqrt(group.data.length));
      const dataWidth = sqrt * stepSize;
      const dataHeight = Math.floor(group.data.length / sqrt) * stepSize;

      const colValue = group.facets[colFeature].displayVal;
      const colPadding = CELL_PADDING * colFeatureVals.indexOf(colValue);
      const colSumToValue = colSizes[colValue].sumTo;

      const rowValue = group.facets[rowFeature].displayVal;
      const rowPadding = CELL_PADDING * rowFeatureVals.indexOf(rowValue);
      const rowSumToValue = rowSizes[rowValue].sumTo;

      const cellHeight = rowSizes[rowValue].size * stepSize;
      const cellHeightDelta = (cellHeight - dataHeight) / 2;
      const cellWidth = colSizes[colValue].size * stepSize;
      const cellWidthDelta = (cellWidth - dataWidth) / 2;

      /**
       * Megaplot has two coordinate systems: world coordinates and pixel
       * coordinates. These systems are additive (world + pixel), allowing
       * simpler positioning of the dots inside of a cell, and the cell inside
       * of the matrix.
       *
       * We use world coordinates to position dots inside a cell, first by
       * computing the dot's position in the grid (derived from its index in
       * `group.data` plus the sum of facet value sizes in that row and column),
       * then by adding an offset to account for differences between this dot
       * grid's size and the size of the containing cell, thereby centering the
       * dot grid in the cell.
       *
       * We use pixel coordinates to add a consistent padding between cells in
       * the matrix, defined by `CELL_PADDING`. Because of the additive nature
       * of these coordinate systems, this padding will always be the same
       * regardless of how zoomed in/out the user is in world space.
       */
      const groupInputs =
          this.colorService.selectedColorOption.name === 'None' ?
          group.data : group.data.sort(sortInputs);

      groupInputs.forEach(({id}, idx) => {
        if (!this.points.has(id)) {
          this.points.set(id, {id, pixel: {x: 0, y: 0}, world: {x: 0, y: 0}});
        }

        const dot: Dot = this.points.get(id)!;
        // Location of this datum in Megaplot's unit square
        const gridX = (Math.floor(idx % sqrt) + colSumToValue) * stepSize;
        const gridY = (Math.floor(idx / sqrt) + rowSumToValue) * stepSize;
        // Final location in world coordinates, translated so the dot-grid
        // appears in the center of the cell.
        dot.world.x = gridX + cellWidthDelta - 0.5;
        dot.world.y = 0.5 - (gridY + cellHeightDelta);
        // Padding between cells in the matrix.
        dot.pixel.x = colPadding;
        dot.pixel.y = rowPadding;
      });
    }

    this.dots = [...this.points.values()];
    this.selections.points?.onInit(bindDot).onUpdate(bindDot);
    this.resetScale();
  }

  /** Groups examples based on the selected row and column features. */
  private groupExamples(groupingFeatures: [string, string]): GroupedExamples {
    const configs: FacetingConfig[] = groupingFeatures
        .filter((featureName) =>
            this.groupService.numericalFeatureNames.includes(featureName))
        .map((featureName) => ({
          featureName,
          method: FacetingMethod.EQUAL_INTERVAL,
        }));
    const numericalBins = this.groupService.numericalFeatureBins(configs);
    return this.groupService.groupExamplesByFeatures(
        numericalBins, this.appState.currentInputData, groupingFeatures);
  }

  /**
   * Computes the `size` (width of a column, height of a row) and the `sumTo`
   * (width or height of the preceeding columns or rows) for each dot group.
   */
  private matrixSizes(groups: GroupedExamples, colFeature: string,
                      rowFeature: string): GroupFeaturesSizes {
    /** Stores size of the largest facet for the column and row features. */
    const sizes: GroupFeaturesSizes = {columns:{}, rows:{}};

    // sorting the
    const keys = Object.keys(groups).sort(keySorter);
    for (const key of keys) {
      const group = groups[key];
      const width = Math.ceil(Math.sqrt(group.data.length));
      const height = Math.ceil(group.data.length / width);

      for (const [feature, {displayVal}] of Object.entries(group.facets)) {
        if (colFeature === rowFeature) {
          if (sizes.columns[displayVal] == null) {
            sizes.columns[displayVal] = {size: width, sumTo: 0};
          } else if (width > sizes.columns[displayVal].size) {
            sizes.columns[displayVal].size = width;
          }

          if (sizes.rows[displayVal] == null) {
            sizes.rows[displayVal] = {size: height, sumTo: 0};
          } else if (height > sizes.rows[displayVal].size) {
            sizes.rows[displayVal].size = height;
          }
        } else {
          const size = colFeature === feature ? width : height;
          const featureDimensions =
              colFeature === feature ? sizes.columns : sizes.rows;

          if (featureDimensions[displayVal] == null) {
            featureDimensions[displayVal] = {size, sumTo: 0};
          } else if (size > featureDimensions[displayVal].size) {
            featureDimensions[displayVal].size = size;
          }
        }
      }
    }

    let columnSum = 0;
    for (const dimensions of Object.values(sizes.columns)) {
      dimensions.sumTo = columnSum;
      columnSum += dimensions.size;
    }

    let rowSum = 0;
    for (const dimensions of Object.values(sizes.rows)) {
      dimensions.sumTo = rowSum;
      rowSum += dimensions.size;
    }
    return sizes;
  }

  /** Binds the labels and dots to the Megaplot scene. */
  private draw() {
    this.selections.columnLabels?.bind(this.columns);
    this.selections.rowLabels?.bind(this.rows);
    this.selections.points?.bind(this.dots);
  }

  /** Resets the scale of the Megaplot Scene. */
  private resetScale() {
    if (this.container == null || this.scene == null) return;
    const {width, height} = this.container.getBoundingClientRect();
    if (width === 0 || height === 0) return;
    const smaller = height < width ? height : width;
    this.scene.scale.x = smaller * HEURISTIC_SCALE_FACTOR;
    this.scene.scale.y = smaller * HEURISTIC_SCALE_FACTOR;
    this.scene.offset.x = Math.ceil(width / 2);
    this.scene.offset.y = Math.floor(height / 2) - CELL_PADDING - TEXT_SIZE;
  }

  /** Resizes the Megaplot Scene canvas and resets the scale. */
  private resize() {
    this.scene?.resize();
    this.resetScale();
  }

  private updateZoom(scaleX: number, scaleY: number) {
    if (this.scene != null) {
      const {x: offsetX, y: offsetY} = this.scene.offset;
      this.scene.scale.x = scaleX;
      this.scene.scale.y = scaleY;
      this.scene.offset.x = offsetX;
      this.scene.offset.y = offsetY;
    }
  }

  private renderControls() {
    const colsChange = (event: Event) => {
      this.colsFeatureIdx = Number((event.target as HTMLInputElement).value);
    };

    const rowsChange = (event: Event) => {
      this.rowsFeatureIdx = Number((event.target as HTMLInputElement).value);
    };

    const zoomChange = (factor: number) => {
      if (this.scene != null) {
        const {x, y} = this.scene.scale;
        this.updateZoom(x * factor, y * factor);
      }
    };

    const dropDownSpecs: Array<[string, number, (e: Event) => void]> = [
      ['Rows:', this.rowsFeatureIdx, rowsChange],
      ['Columns:', this.colsFeatureIdx, colsChange]
    ];

    // clang-format off
    return html`<div class="dive-controls">
      <div class="feature-selectors">
        ${dropDownSpecs.map(([label, index, onChange]) => html`
        <div class="dropdown-holder">
          <label class="dropdown-label">${label}</label>
          <select class="dropdown limit-width" @change=${onChange}
                  title=${this.groupService.denseFeatureNames[index]}>
            ${this.groupService.denseFeatureNames.map((feature, i) => html`
              <option value=${i} ?selected=${index === i}>
                ${feature}
              </option>`)}
          </select>
        </div>`)}
      </div>
      <div class="view-controls">
        <span class="material-icon-outlined icon-button" title="Zoom in"
              @click=${() =>{zoomChange(ZOOM_IN_FACTOR);}}>
          zoom_in
        </span>
        <span class="material-icon-outlined icon-button" title="Zoom out"
              @click=${() =>{zoomChange(ZOOM_OUT_FACTOR);}}>
          zoom_out
        </span>
        <span class="material-icon-outlined icon-button" title="Reset view"
              @click=${() =>{this.resetScale();}}>
          view_in_ar
        </span>
      </div>
    </div>`;
    // clang-format on
  }

  private renderFooter() {
    const [min] = this.colorService.selectedColorOption.scale.domain();
    const legendType = typeof min === 'number' ?  LegendType.SEQUENTIAL :
                                                  LegendType.CATEGORICAL;

    // clang-format off
    return html`<color-legend legendType=${legendType}
      selectedColorName=${this.colorService.selectedColorOption.name}
      .scale=${this.colorService.selectedColorOption.scale}>
    </color-legend>`;
    // clang-format on
  }

  // Purposely overridding render() as opposed to renderImpl() as Dive is a
  // special case where the render pass here is just a set up for the containers
  // and the true rendering happens in reactions.
  override render() {
    const select = (event: MouseEvent) => {
      const {offsetX: x, offsetY: y} = event;
      const selected = this.selections.points?.hitTest({x, y});
      if (selected?.length) {
        const primary = selected[0].id;
        const ids = selected.map(d => d.id);

        // Hold down Alt/Option key to pin a datapoint.
        if (event.altKey) {
          this.pinnedSelectionService.selectIds([]);
          this.pinnedSelectionService.setPrimarySelection(primary);
          return;
        }

        // Hold down Ctrl or Cmd to preserve current selection.
        if (event.metaKey || event.ctrlKey) {
          ids.unshift(...this.selectionService.selectedIds);
        }

        this.selectionService.selectIds(ids);
        this.selectionService.setPrimarySelection(primary);
      } else {
        this.selectionService.selectIds([]);
      }
    };

    const mousedown = (event: MouseEvent) => {
      this.mouseDown = {x: event.offsetX, y: event.offsetY};
    };

    const mousemove = (event: MouseEvent) => {
      const {offsetX: x, offsetY: y} = event;

      if (this.mouseDown != null) {   // This is a pan interaction
        if (this.scene == null) return;
        this.focusService.clearFocus();
        this.scene.offset.x -= this.mouseDown.x - x;
        this.scene.offset.y -= this.mouseDown.y - y;
        this.mouseDown = {x, y};
      } else {  // This is a hover interaction
        const hovered = this.selections.points?.hitTest({x, y});
        if (hovered?.length) {
          this.focusService.setFocusedDatapoint(hovered[0].id);
        } else {
          this.focusService.clearFocus();
        }
      }
    };

    const mouseup = (event: MouseEvent) => {
      this.mouseDown = undefined;
    };

    const wheelZoom = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();

      if (this.scene != null) {
        const {x, y} = this.scene.scale;
        const factor = event.deltaY * ZOOM_WHEEL_FACTOR;
        this.updateZoom(x + factor, y + factor);
      }
    };

    // clang-format off
    return html`<div class="module-container">
      <div class='module-toolbar'>${this.renderControls()}</div>
      <div class="module-results-area">
        <div class="scene" @click=${select} @mousedown=${mousedown}
             @mousemove=${mousemove} @mouseout=${mouseup} @mouseup=${mouseup}
             @wheel=${wheelZoom}></div>
      </div>
      <div class="module-footer">${this.renderFooter()}</div>
    </div>`;
    // clang-format on
  }

  override firstUpdated() {
    if (this.container == null) return;
    this.scene = new Scene({container: this.container,
                            defaultTransitionTimeMs: 300});

    this.selections.columnLabels = this.scene.createTextSelection<Label>();
    this.selections.columnLabels.align(() => 'center')
        .verticalAlign(() => 'bottom').text((label: Label) => label.text)
        .onInit(bindLabel).onUpdate(bindLabel).onExit(spriteExitHandler);

    this.selections.rowLabels = this.scene.createTextSelection<Label>();
    this.selections.rowLabels.align(() => 'right')
        .verticalAlign(() => 'middle').text((label: Label) => label.text)
        .onInit(bindLabel).onUpdate(bindLabel).onExit(spriteExitHandler);

    this.selections.points = this.scene.createSelection<Dot>();
    this.selections.points.onExit(spriteExitHandler);

    this.resizeObserver.observe(this.container);

    this.reactImmediately(() => this.appState.currentDataset, () => {
      // Dataset changed, clear out the existing visualization
      this.selections.columnLabels?.clear();
      this.selections.rowLabels?.clear();
      this.selections.points?.clear();
      this.points = new Map<string, Dot>();
    });

    const configChanges = () => [
        this.colsFeatureIdx, this.rowsFeatureIdx, this.dataService.dataVals,
        this.colorService.selectedColorOption];
    this.reactImmediately(configChanges, () => {
      const colorFeature = this.colorService.selectedColorOption.name;
      if (colorFeature !== 'None') {
        for (const {id} of this.appState.currentInputData) {
          this.colorValueMap.set(id, this.dataService.getVal(id, colorFeature));
        }
      }

      this.configure();
      this.draw();
    });

    const selectionChanges = () => [
        this.selectionService.selectedIds,
        this.selectionService.primarySelectedId,
        this.pinnedSelectionService.primarySelectedId,
        this.focusService.focusData?.datapointId];
    this.reactImmediately(selectionChanges, () => {this.draw();});
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'dive-module': DiveModule;
  }
}
