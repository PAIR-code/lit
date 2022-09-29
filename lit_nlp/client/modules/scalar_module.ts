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

import '../elements/expansion_panel';

// tslint:disable:no-new-decorators
// taze: ResizeObserver from //third_party/javascript/typings/resize_observer_browser
import * as d3 from 'd3';
import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {Scene, Selection, SpriteView} from 'megaplot';
import {computed, observable} from 'mobx';
// tslint:disable-next-line:ban-module-namespace-object-escape
const seedrandom = require('seedrandom');  // from //third_party/javascript/typings/seedrandom:bundle

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {ThresholdChange} from '../elements/threshold_slider';
import {colorToRGB, getBrandColor} from '../lib/colors';
import {MulticlassPreds, Scalar} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatForDisplay, IndexedInput, ModelInfoMap, ModelSpec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, getThresholdFromMargin} from '../lib/utils';
import {CalculatedColumnType, CLASSIFICATION_SOURCE_PREFIX, REGRESSION_SOURCE_PREFIX, SCALAR_SOURCE_PREFIX} from '../services/data_service';
import {ClassificationService, ColorService, DataService, FocusService, GroupService, SelectionService} from '../services/services';

import {styles} from './scalar_module.css';

/** The maximum number of scatterplots to render on page load. */
export const MAX_DEFAULT_PLOTS = 2;

const CANVAS_PADDING = 8;
const DEFAULT_BORDER_WIDTH = 2;
const DEFAULT_LINE_COLOR = '#cccccc';
const DEFAULT_SCENE_PARAMS = {defaultTransitionTimeMs: 0};
const RGBA_CYEA_700 = colorToRGB(getBrandColor('cyea', '700').color);
const RGBA_MAGE_400 = colorToRGB(getBrandColor('mage', '400').color);
const RGBA_MAGE_700 = colorToRGB(getBrandColor('mage', '700').color);
const RGBA_WHITE = colorToRGB('white');
const SPRITE_SIZE_LG = 12 + DEFAULT_BORDER_WIDTH;
const SPRITE_SIZE_MD = 10 + DEFAULT_BORDER_WIDTH;
const SPRITE_SIZE_SM = 8 + DEFAULT_BORDER_WIDTH;
const X_LABELS_PADDING = 12;
const Y_LABELS_PADDING = 40;


/** Indexed scalars for an id, inclusive of model input and output scalars. */
interface IndexedScalars {
  id: string;
  data:{
    // Values in this structure can be undefined if the call to
    // DataService.getVal(2) in this.updatePredictions() occurs before the
    // async calls in DataService have returned the scalar values for model's
    // scalar predictions, regressions, and classifications. If these calls have
    // not yet returned, DataService.getVal(2) will try to get the value from
    // the Dataset, but since the key being used is from the model's output spec
    // the function will return undefined. These eventually get sorted out via
    // updates, but can cause hidden/frustrating errors in D3.attr(2) calls if
    // not handled appropriately.
    [key: string]: number | number[] | undefined
  };
  rngY: number;
}

interface PlotInfo {
  hidden: boolean;
  /** The prediction key in output spec. */
  key: string;
  /** The label of interest in the vocab. */
  label?: string;
  /** A MegaPlot Selection that binds all data to the Scene. */
  points?: Selection<IndexedScalars>;
  /** The MegaPlot Scene into which the scatterplot is rendered. */
  scene?: Scene;
  xScale?: d3.ScaleLinear<number, number>;
  yScale?: d3.ScaleLinear<number, number>;
  brush?: d3.BrushBehavior<unknown>;
}

/**
 * A LIT module that visualizes prediction scores and other scalar values.
 */
@customElement('scalar-module')
export class ScalarModule extends LitModule {
  static override title = 'Scalars';
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`<scalar-module model=${model} .shouldReact=${shouldReact}
                selectionServiceIndex=${selectionServiceIndex}>
              </scalar-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap) {
    return doesOutputSpecContain(modelSpecs, [Scalar, MulticlassPreds]);
  }

  private readonly colorService = app.getService(ColorService);
  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly groupService = app.getService(GroupService);
  private readonly focusService = app.getService(FocusService);
  private readonly dataService = app.getService(DataService);
  private readonly pinnedSelectionService =
      app.getService(SelectionService, 'pinned');

  private readonly plots = new Map<string, PlotInfo>();
  private readonly resizeObserver =
      new ResizeObserver(() => {this.resizePlots();});

  private numPlotsRendered: number = 0;
  @observable private preds: IndexedScalars[] = [];

  @computed get datasetSize(): number {
    return this.appState.metadata.datasets[this.appState.currentDataset].size;
  }

  @computed private get scalarColumnsToPlot() {
    return this.groupService.numericalFeatureNames.filter(feat => {
      const col = this.dataService.getColumnInfo(feat);
      if (col == null) {
        return true;  // Col will be null for input fields
      } else if (col.source.includes(CLASSIFICATION_SOURCE_PREFIX) ||
                 col.source.includes(REGRESSION_SOURCE_PREFIX) ||
                 col.source.includes(SCALAR_SOURCE_PREFIX)) {
        return col.source.includes(this.model);
      } else {
        return true;
      }
    });
  }

  @computed private get classificationKeys() {
    const {output} = this.appState.getModelSpec(this.model);
    return findSpecKeys(output, MulticlassPreds);
  }

  private containerSelector(id: string) {
    return `div.scatterplot[data-id="${id}"]`;
  }

  override firstUpdated() {
    const getDataChanges = () => [
      this.appState.currentInputData,
      // TODO(b/156100081): Reacting to this.dataService.dataVals incurs a
      // pretty substantial reaction overhead penalty at 100k datapoints. This
      // would be better if we observed changes to the number of columns/rows in
      // the DataService instead.
      this.dataService.dataVals,
      this.scalarColumnsToPlot
    ];
    this.reactImmediately(getDataChanges, () => {
      for (const info of this.plots.values()) {info.points?.clear();}
      this.updatePredictions(this.appState.currentInputData);
    });

    const rebindChanges = () => [
      this.colorService.selectedColorOption,
      this.preds,
      this.selectionService.selectedIds,
      this.selectionService.primarySelectedId,
      this.pinnedSelectionService.primarySelectedId,
      this.focusService.focusData?.datapointId
    ];
    this.react(rebindChanges, () => {this.updatePlots();});

    const container = this.shadowRoot!.getElementById('container')!;
    this.resizeObserver.observe(container);
  }

  override updated() {this.updatePlots();}

  /**
   * Get predictions from the backend for all input data and display by
   * prediction score in the plot.
   */
  private async updatePredictions(currentInputData?: IndexedInput[]) {
    if (currentInputData == null) {return;}

    // tslint:disable-next-line:no-any ban-module-namespace-object-escape
    const rng = seedrandom('lit');
    const preds: IndexedScalars[] = [];

    for (const {id} of currentInputData) {
      const pred: IndexedScalars = {id, data: {}, rngY: rng()};
      for (const key of this.classificationKeys) {
        const column = this.dataService.getColumnName(this.model, key);
        pred.data[key] = this.dataService.getVal(id, column);
      }

      for (const scalarKey of this.scalarColumnsToPlot) {
        pred.data[scalarKey] = this.dataService.getVal(id, scalarKey);
      }
      preds.push(pred);
    }

    this.preds = preds;
  }

  /**
   * Returns the scale function for the scatter plot's x axis, for a given
   * key. If the key is for regression, we set the score range to be between the
   * min and max values of the regression scores.
   */
  private getXScale(key: string) {
    let scoreRange = [0, 1];

    const {output} = this.appState.getModelSpec(this.model);
    if (output[key] instanceof Scalar) {
      const scalarValues = this.preds.map((pred) => pred.data[key]) as number[];
      scoreRange = [Math.min(...scalarValues), Math.max(...scalarValues)];
      // If the range is 0 (all values are identical, then artificially increase
      // the range so that an X-axis is properly displayed.
      if (scoreRange[0] === scoreRange[1]) {
        scoreRange[0] = scoreRange[0] - .1;
        scoreRange[1] = scoreRange[1] + .1;
      }
    } else if (this.scalarColumnsToPlot.indexOf(key) !== -1) {
      scoreRange = this.groupService.numericalFeatureRanges[key];
    }

    return d3.scaleLinear().domain(scoreRange).range([0, 1]);
  }

  /**
   * Returns the scale function for the scatter plot's y axis.
   */
  private getYScale(key: string, isRegression: boolean, errorColumn: string) {
    const scale = d3.scaleLinear().domain([0, 1]).range([0, 1]);

    if (isRegression) {
      const values = this.dataService.getColumn(errorColumn);
      const range = d3.extent(values);
      if (range != null && !range.some(isNaN)) {
        // Make the domain symmetric around 0
        const largest = Math.max(...(range as number[]).map(Math.abs));
        scale.domain([-largest, largest]);
      }
    }

    return scale;   // Regression output field
  }

  private getValue(preds: IndexedScalars, spec: ModelSpec, key: string,
                   label: string): number | undefined {
    // If for a multiclass prediction and the DataService has loaded the
    // classification results, return the label score from the array.
    if (spec.output[key] instanceof MulticlassPreds) {
      const {vocab} = spec.output[key] as MulticlassPreds;
      const index = vocab!.indexOf(label);
      const classPreds = preds.data[key];
      if (Array.isArray(classPreds)) return classPreds[index];
    }
    // Otherwise, return the value of data[key], which may be undefined if the
    // DataService async calls are still pending.
    return preds.data[key] as number | undefined;
  }

  /** Sets up the scatterplot using MegaPlot (points) and D3 (axes). */
  private setupPlot(info: PlotInfo, container: HTMLElement) {
    const {key, label} = info;
    const axesDiv = container.querySelector<HTMLDivElement>('.axes')!;
    const sceneDiv = container.querySelector<HTMLDivElement>('.scene')!;
    const {width, height} = sceneDiv.getBoundingClientRect();

    // Clear any existing content
    axesDiv.textContent = '';
    sceneDiv.textContent = '';

    // Determine if this is a RegressionScore column
    const errorColName = `${key}:${CalculatedColumnType.ERROR}`;
    const errFeatInfo = this.dataService.getColumnInfo(errorColName);
    const isRegression = errFeatInfo != null &&
        errFeatInfo.source.includes(REGRESSION_SOURCE_PREFIX);

    // X and Y scales and accessors
    const xScale = info.xScale =
        this.getXScale(key).range([0, width - 2 * CANVAS_PADDING]);
    const yScale = info.yScale =
        this.getYScale(key, isRegression, errorColName).range(
          [CANVAS_PADDING, height - CANVAS_PADDING]);

    // Add the axes with D3
    d3.select(axesDiv).style('width', width).style('height', height);
    const axesSVG = d3.select(axesDiv).append<SVGSVGElement>('svg')
                      .style('width', width + Y_LABELS_PADDING)
                      .style('height', height + X_LABELS_PADDING);

    axesSVG.append('g')
           .attr('id', 'xAxis')
           .attr('transform', `translate(40, ${height - CANVAS_PADDING})`)
           .call(d3.axisBottom(info.xScale));

    axesSVG.append('g')
           .attr('id', 'yAxis')
           .attr('transform', `translate(40, 0)`)
           .call(d3.axisLeft(info.yScale).ticks(isRegression ? 5 : 0 ));

    const lines = axesSVG.append('g')
                         .attr('id', 'lines')
                         .attr('transform', `translate(40, 0)`);

    const [xMin, xMax] = info.xScale.range();
    const [yMin, yMax] = info.yScale.range();

    if (isRegression) {
      const halfHeight = (yMax - yMin) / 2 + yMin;
      lines.append('line')
           .attr('id', 'regression-line')
           .attr('x1', xMin)
           .attr('y1', halfHeight)
           .attr('x2', xMax)
           .attr('y2', halfHeight)
           .style('stroke', DEFAULT_LINE_COLOR);
    }

    const fieldSpec = this.appState.getModelSpec(this.model).output[key];
    if (fieldSpec instanceof MulticlassPreds && fieldSpec.null_idx != null &&
        fieldSpec.vocab.length !== 2) {
      const margin = this.classificationService.getMargin(this.model, key);
      const threshold = xScale(getThresholdFromMargin(margin));
      lines.append('line')
            .attr('id', 'threshold-line')
            .attr('x1', threshold)
            .attr('y1', yMin)
            .attr('x2', threshold)
            .attr('y2', yMax)
            .style('stroke', DEFAULT_LINE_COLOR);
    }

    const brush = info.brush = isRegression ? d3.brush() : d3.brushX();
    const brushGroup = axesSVG.append('g')
        .attr('id', 'brushGroup')
        .attr('transform', `translate(40, 0)`)
        .on('mouseenter', () => {this.focusService.clearFocus();});

    brush.extent([[xMin, 0], [xMax, yMax + X_LABELS_PADDING]])
        .on('start end', () => {
          const bounds = d3.event.selection;
          if (!d3.event.sourceEvent || !bounds?.length) return;

          const hasYDimension = Array.isArray(bounds[0]);
          const [x, x2] = (hasYDimension ? [bounds[0][0], bounds[1][0]] :
                                           bounds) as number[];
          const [y, y2] = (hasYDimension ? [bounds[0][1], bounds[1][1]] :
                                           [yMin, yMax]) as number[];

          // Clear selection if brush size is imperceptible along either axis
          if (x2 - x < window.devicePixelRatio ||
              y2 - y < window.devicePixelRatio) {
            this.selectionService.selectIds([]);
          } else {
            const ids = info.points?.hitTest(
                {x, y, width: (x2 - x), height: (y2 - y)}).map((p) => p.id);
            if (ids != null) this.selectionService.selectIds(ids);
          }

          brushGroup.call(brush.move, null);
        });

    brushGroup.call(brush);

    // Render the scatterplot with MegaPlot
    info.scene = new Scene({
      container: sceneDiv,
      desiredSpriteCapacity: this.datasetSize,
      ...DEFAULT_SCENE_PARAMS
    });
    info.scene.scale.x = 1;
    info.scene.scale.y = 1;
    info.scene.offset.x = CANVAS_PADDING;
    info.scene.offset.y = height;

    info.points = info.scene.createSelection<IndexedScalars>()
        .onExit((sprite: SpriteView) => {sprite.SizePixel = 0;})
        .onBind((sprite: SpriteView, pred: IndexedScalars) => {
          const modelSpec = this.appState.getModelSpec(this.model);

          // TODO(b/243566359): Normalizing position values in Megaplot's world
          // coordinates (i.e.,the range [-.5, .5]) would make zooming easy.
          const xValue = this.getValue(pred, modelSpec, key, label || '');
          const xScaledValue = xValue != null ? xScale(xValue) : NaN;
          sprite.PositionWorldX = isNaN(xScaledValue) ? 0 : xScaledValue;

          const yValue = isRegression ?
              this.dataService.getVal(pred.id, errorColName) : pred.rngY;
          const yPosition = yValue != null ? yScale(yValue) : NaN;
          sprite.PositionWorldY = isNaN(yPosition) ? yMax : yPosition;

          const isHovered =
              this.focusService.focusData?.datapointId === pred.id;
          const isPinned =
              this.pinnedSelectionService.primarySelectedId === pred.id;
          const isPrimary =
              this.selectionService.primarySelectedId === pred.id;
          const isSelected =
              this.selectionService.selectedIds.includes(pred.id) &&
              !(isHovered || isPinned || isPrimary);
          const isSpecial = isHovered || isPinned || isPrimary || isSelected;

          const indexedInput = this.appState.getCurrentInputDataById(pred.id);
          const colorString = this.colorService.getDatapointColor(indexedInput);
          const color = colorToRGB(colorString);
          sprite.BorderRadiusPixel = DEFAULT_BORDER_WIDTH;
          sprite.BorderColor = isHovered ? RGBA_MAGE_400 :
                               isPinned ? RGBA_MAGE_700 :
                               isPrimary ? RGBA_CYEA_700 :
                               isSelected ? RGBA_WHITE : color;
          sprite.BorderColorOpacity = isSpecial ? 1 : 0.25;
          sprite.FillColor =  (isHovered || isPinned) ? RGBA_MAGE_400 : color;
          sprite.FillColorOpacity = isSpecial ? 1 : 0.25;
          sprite.OrderZ = !isSpecial ? 0 : isHovered ? 1 : isSelected ? .5 : .8;
          sprite.Sides = 1;
          sprite.SizeWorld = !isSpecial ? SPRITE_SIZE_SM :
                              isSelected ? SPRITE_SIZE_MD : SPRITE_SIZE_LG;
        });
  }

  /** Binds the predictions to the available MegaPlot Selection and Scene. */
  private updatePlots() {
    for (const [id, info] of this.plots.entries()) {
      const container = this.renderRoot.querySelector<HTMLDivElement>(
          this.containerSelector(id));
      const {hidden, scene} = info;
      // Don't render/update if hidden or no container
      if (hidden || container == null) continue;
      const {width, height} = container.getBoundingClientRect();
      // Don't render/update if container is invisible
      if (width === 0 || height === 0) continue;
      if (scene == null) this.setupPlot(info, container);
      const {points} = info;
      if (points != null) points.bind(this.preds);
    }
  }

  /** Updates the Megaplot Scenes and D3 axes for each plot in this module. */
  private resizePlots() {
    // All Scalars plots are the same height (100px), so the only axis that can
    // be resized is the X axis.
    for (const [id, info] of this.plots.entries()) {
      const container = this.renderRoot.querySelector<HTMLDivElement>(
          this.containerSelector(id));
      if (container == null || info.hidden) continue;

      const {xScale, yScale, brush, scene, key, points} = info;
      if (xScale == null || yScale == null || brush == null || scene == null ||
          key == null || points == null) continue;

      // Update the xScale range  width of div.axes
      const sceneDiv = container.querySelector<HTMLDivElement>('.scene')!;
      const {width} = sceneDiv.getBoundingClientRect();

      xScale.range([0, width - 2 * CANVAS_PADDING]);
      const [xMin, xMax] = xScale.range();
      const yMax = yScale.range()[1];

      // Then update the SVG components -- axes, threshold line, regression line
      const axesDiv = container.querySelector<HTMLDivElement>('.axes')!;
      const axesSVG = d3.select(axesDiv).select<SVGSVGElement>('svg');
      axesSVG.style('width', width + Y_LABELS_PADDING);
      axesSVG.select<SVGGElement>('g#xAxis').call(d3.axisBottom(xScale));
      const lines = axesSVG.select<SVGGElement>('g#lines');

      const regressionLine = lines.select<SVGLineElement>('#regression-line');
      if (!regressionLine.empty()) {
        regressionLine.attr('x1', xMin).attr('x2', xMax);
      }

      const thresholdLine = lines.select<SVGLineElement>('#threshold-line');
      if (!thresholdLine.empty()) {
        const margin = this.classificationService.getMargin(this.model, key);
        const threshold = xScale(getThresholdFromMargin(margin));
        thresholdLine.attr('x1', threshold).attr('x2', threshold);
        console.log(`updated threshold line for ${id}`);
      }

      brush.extent([[xMin, 0], [xMax, yMax + X_LABELS_PADDING]]);

      // Finally, update the Megaplot scene
      const {x, y} = scene.offset;
      scene.resize(scene.offset);
      scene.offset.x = x;
      scene.offset.y = y;
      // TODO(b/243566359): If positions are normalized in Megaplot's world
      // coordinates we can set scene.scale instead of re-binding.
      points.bind(this.preds);
    }
  }

  // Purposely overridding render() as opposed to renderImpl() as scalars are a
  // special case where the render pass here is just a set up for the containers
  // and the true rendering happens in reactions, due to the nature of this
  // module.
  override render() {
    this.numPlotsRendered = 0;
    const colorOption = this.colorService.selectedColorOption;
    const domain = colorOption.scale.domain();
    const legendType = typeof domain[0] === 'number' ? LegendType.SEQUENTIAL :
                                                       LegendType.CATEGORICAL;

    // clang-format off
    return html`<div class="module-container">
      <div class="module-results-area">
        <div id='container'>
          ${this.classificationKeys.map(key =>
            this.renderClassificationGroup(key))}
          ${this.scalarColumnsToPlot.map(key => this.renderPlot(key, ''))}
        </div>
      </div>
      <div class="module-footer">
        <color-legend legendType=${legendType} .scale=${colorOption.scale}
          selectedColorName=${colorOption.name}>
        </color-legend>
      </div>
    </div>`;
    // clang-format on
  }

  private renderClassificationGroup(key: string) {
    const spec = this.appState.getModelSpec(this.model);
    const fieldSpec = spec.output[key];
    if (!(fieldSpec instanceof MulticlassPreds)) return;
    const {vocab, null_idx} = fieldSpec;

    // In the binary classification case, only render one plot that
    // displays the positive class.
    if (vocab?.length === 2 && null_idx != null) {
      return html`${this.renderPlot(key, vocab[1 - null_idx])}`;
    }

    // Otherwise, return one plot per label in the multiclass case.
    // clang-format off
    return html`${vocab.map(label => this.renderPlot(key, label))}
                ${null_idx != null ? this.renderMarginSlider(key) : null}`;
    // clang-format on
  }

  private renderPlot(key: string, label: string) {
    this.numPlotsRendered += 1;
    const id = label ? `${key}:${label}` : key;
    const {primarySelectedId} = this.selectionService;
    const primaryPreds = this.preds.find(pred => pred.id === primarySelectedId);

    let selectedValue;
    if (primaryPreds != null) {
      const modelSpec = this.appState.getModelSpec(this.model);
      const primaryValue =
          this.getValue(primaryPreds, modelSpec, key, label);
      selectedValue = `Value: ${formatForDisplay(primaryValue)}`;
    }

    const plotLabel = `${id}${selectedValue ? ` - ${selectedValue}` : ''}`;

    if (!this.plots.has(id)) {
      const hidden = this.numPlotsRendered > MAX_DEFAULT_PLOTS;
      this.plots.set(id, {key, label, hidden});
    }
    const info = this.plots.get(id)!;

    /**
     * Rebases a MouseEvent's X and Y coordinates to be relative to the origin
     * of the `div.scene` associated with this plot.
     *
     * Megaplot expects coordinates sent to `.hitTest()` to be defined in the
     * coordinate system of the canvas element it creates. Since the `clientX`
     * and `clientY` properties of a `MouseEvent` are relative to the DOM
     * content, we need to rebase them so they are coordinates in the canvas's
     * coordinate system, i.e., relative to `div.scene`'s `top` and `left` as
     * the canvas fills its parent.
     */
    const rebase = (event: Pick<MouseEvent, 'clientX' | 'clientY'>) => {
      const {clientX, clientY} = event;
      const selector = `div.scatterplot[data-id="${id}"] div.scene`;
      const scene = this.renderRoot.querySelector<HTMLDivElement>(selector);
      const {top, left} = scene!.getBoundingClientRect();
      return {x: clientX - left, y: clientY - top};
    };

    const select = (event: MouseEvent) => {
      const {x, y} = rebase(event);
      const selected = info.points?.hitTest({x, y});
      if (selected?.length) {
        this.selectionService.setPrimarySelection(selected[0].id);
      }
    };

    const hover = (event: MouseEvent) => {
      const {x, y} = rebase(event);
      const hovered = info.points?.hitTest({x, y});
      if (hovered?.length) {
        this.focusService.setFocusedDatapoint(hovered[0].id);
      } else {
        this.focusService.clearFocus();
      }
    };

    const toggleHidden = () => {
      info.hidden = !info.hidden;
      if (!info.scene) this.requestUpdate();
    };

    // clang-format off
    return html`<div class='plot-holder'>
      <expansion-panel  .label=${plotLabel} ?expanded=${!info.hidden}
                        @expansion-toggle=${toggleHidden}>
        <div  class="scatterplot" data-id=${id} @mousemove=${hover}
              @click=${select}>
          <div class="scene"></div>
          <div class="axes"></div>
        </div>
      </expansion-panel>
    </div>`;
    // clang-format on
  }

  private renderMarginSlider(key: string) {
    const margin = this.classificationService.getMargin(this.model, key);
    const callback = (e: CustomEvent<ThresholdChange>) => {
      this.classificationService.setMargin(this.model, key, e.detail.margin);
    };
    return html`<threshold-slider label=${key} ?isThreshold=${false}
      .margin=${margin} ?showControls=${true} @threshold-changed=${callback}>
    </threshold-slider>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'scalar-module': ScalarModule;
  }
}
