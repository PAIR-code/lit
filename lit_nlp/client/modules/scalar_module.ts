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
import {html, TemplateResult} from 'lit';
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
import {MulticlassPreds, RegressionScore, Scalar} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatForDisplay, IndexedInput, ModelInfoMap} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, getThresholdFromMargin} from '../lib/utils';
import {CalculatedColumnType, REGRESSION_SOURCE_PREFIX} from '../services/data_service';
import {ClassificationService, ColorService, DataService, FocusService, SelectionService} from '../services/services';

import {styles} from './scalar_module.css';

/** The maximum number of scatterplots to render on page load. */
export const MAX_DEFAULT_PLOTS = 2;

const CANVAS_PADDING = 8;
const DEFAULT_BORDER_WIDTH = 2;
const DEFAULT_LINE_COLOR = getBrandColor('neutral', '600').color;
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
  rngY: number;
}

interface PlotInfo {
  hidden: boolean;
  /** The field containing a scalar-like value. */
  key: string;
  /** The model making the prediction. */
  model?: string;
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
  static override duplicateForModelComparison = false;
  static override title = 'Scalars';
  static override referenceURL =
      'https://github.com/PAIR-code/lit/wiki/components.md#scalar-plots';
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`<scalar-module selectionServiceIndex=${selectionServiceIndex}
                .shouldReact=${shouldReact} ></scalar-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap) {
    return doesOutputSpecContain(modelSpecs, [Scalar, MulticlassPreds]);
  }

  private readonly colorService = app.getService(ColorService);
  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly focusService = app.getService(FocusService);
  private readonly dataService = app.getService(DataService);
  private readonly pinnedSelectionService =
      app.getService(SelectionService, 'pinned');

  private readonly plots = new Map<string, PlotInfo>();
  private readonly resizeObserver =
      new ResizeObserver(() => {this.resizePlots();});

  private numPlotsRendered: number = 0;
  @observable private preds: IndexedScalars[] = [];

  @computed get datasetScalarKeys(): string[] {
    return findSpecKeys(this.appState.currentDatasetSpec, Scalar);
  }

  @computed get datasetSize(): number {
    return this.appState.metadata.datasets[this.appState.currentDataset].size;
  }

  private containerSelector(id: string) {
    return `div.scatterplot[data-id="${id}"]`;
  }

  override firstUpdated() {
    this.reactImmediately(
        () => this.classificationService.allMarginSettings,
        () => {
          for (const [id, {model, key, xScale}] of this.plots.entries()) {
            if (model == null) continue;
            const {output} = this.appState.getModelSpec(model);
            const fieldSpec = output[key];
            if (!(fieldSpec instanceof MulticlassPreds)) continue;
            const container = this.renderRoot.querySelector<HTMLDivElement>(
                this.containerSelector(id))!;
            const thresholdLine = d3.select(container)
                                    .select<SVGLineElement>('#threshold-line');
            if (!thresholdLine.empty() && xScale != null) {
              const margin = this.classificationService.getMargin(model, key);
              const threshold = xScale(getThresholdFromMargin(margin));
              thresholdLine.attr('x1', threshold).attr('x2', threshold);
            }
          }
        });

    const getDataChanges = () => [
      this.appState.currentInputData,
      // TODO(b/156100081): Reacting to this.dataService.dataVals incurs a
      // pretty substantial reaction overhead penalty at 100k datapoints. This
      // would be better if we observed changes to the number of columns/rows in
      // the DataService instead.
      this.dataService.dataVals
    ];
    this.reactImmediately(getDataChanges, () => {
      for (const info of this.plots.values()) {
        info.points?.clear();
        delete info.points;
        delete info.scene;
      }
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
    this.preds = currentInputData.map(({id}) => ({id, rngY: rng()}));
  }

  /**
   * Returns the scale function for the scatter plot's x axis, for a given
   * key. If the key is for regression, we set the score range to be between the
   * min and max values of the regression scores.
   */
  private getXScale(key: string, model?: string) {
    let scoreRange: [number, number];

    if (model == null) {  // Then this is a key from the dataset.
      const values = this.dataService.getColumn(key) as number[];
      scoreRange = [Math.min(...values), Math.max(...values)];
    } else {              // Otherwise this key is from a model
      const {output} = this.appState.getModelSpec(model);
      const column = this.dataService.getColumnName(model, key);
      const values = this.dataService.getColumn(column) as number[];
      const fieldSpec = output[key];
      // Models can have Scalar our MulticlassPreds outputs. Use the fixed range
      // [0, 1] for MulticlassPreds, otherwise get the min and max values.
      scoreRange = fieldSpec instanceof MulticlassPreds ?
          [0, 1] : [Math.min(...values), Math.max(...values)];
    }

    return d3.scaleLinear().domain(scoreRange).range([0, 1]);
  }

  /**
   * Returns the scale function for the scatter plot's y axis.
   */
  private getYScale(isRegression: boolean, errorColumn: string) {
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

  private getValue(id: string, key: string, model?:string,
                   label?: string): number | undefined {
    // If the field is in the dataset, return that value from the DataService
    if (key in this.appState.currentDatasetSpec) {
      return this.dataService.getVal(id, key);
    }

    // If model is falsy return undefined
    if (model == null) {return undefined;}

    // Otherwise it's from a model.
    const spec = this.appState.getModelSpec(model);
    const columnName = this.dataService.getColumnName(model, key);
    const fieldSpec = spec.output[key];

    // If a MulticlassPreds and the DataService has loaded the
    // classification results, return the label score from the array.
    if (fieldSpec instanceof MulticlassPreds) {
      if (!label) {return undefined;}
      const {vocab} = fieldSpec;
      const index = vocab.indexOf(label);
      const classPreds = this.dataService.getVal(id, columnName);
      if (Array.isArray(classPreds)) {
        return classPreds[index];
      } else {
        return undefined;
      }
    }
    // Otherwise, return the value stored in the DataService, which may be
    // undefined if the async calls are still pending.
    return this.dataService.getVal(id, columnName) as number | undefined;
  }

  /** Sets up the scatterplot using MegaPlot (points) and D3 (axes). */
  private setupPlot(info: PlotInfo, container: HTMLElement) {
    const {model, key, label} = info;
    const axesDiv = container.querySelector<HTMLDivElement>('.axes')!;
    const sceneDiv = container.querySelector<HTMLDivElement>('.scene')!;
    const {width, height} = sceneDiv.getBoundingClientRect();

    // Clear any existing content
    axesDiv.textContent = '';
    sceneDiv.textContent = '';

    // Determine if this is a RegressionScore column
    const errorColName =this.dataService.getColumnName(
        model || '', key, CalculatedColumnType.ERROR);
    const errFeatInfo = this.dataService.getColumnInfo(errorColName);
    const isRegression = errFeatInfo != null &&
        errFeatInfo.source.includes(REGRESSION_SOURCE_PREFIX);

    // X and Y scales for Megaplot world space
    const xScale = this.getXScale(key, model);
    const yScale = this.getYScale(isRegression, errorColName);

    // X and Y scales for D3 pixel space
    info.xScale = this.getXScale(key, model)
                      .range([0, width - 2 * CANVAS_PADDING]);
    // D3 needs to invert the Y axis domain because D3 goes top-down and
    // Megaplot goes bottom up through their ranges.
    info.yScale = d3.scaleLinear()
                      .domain([yScale.domain()[1], yScale.domain()[0]])
                      .range([CANVAS_PADDING, height - CANVAS_PADDING]);

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

    if (model != null) {
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

      const fieldSpec = this.appState.getModelSpec(model).output[key];
      if (fieldSpec instanceof MulticlassPreds && fieldSpec.null_idx != null) {
        const margin = this.classificationService.getMargin(model, key);
        const threshold = info.xScale(getThresholdFromMargin(margin));
        lines.append('line')
              .attr('id', 'threshold-line')
              .attr('x1', threshold)
              .attr('y1', yMin)
              .attr('x2', threshold)
              .attr('y2', yMax)
              .style('stroke', DEFAULT_LINE_COLOR);
      }
    }

    // Render the scatterplot with MegaPlot
    info.scene = new Scene({
      container: sceneDiv,
      desiredSpriteCapacity: this.datasetSize,
      ...DEFAULT_SCENE_PARAMS
    });
    info.scene.scale.x = width - 2 * CANVAS_PADDING;
    info.scene.scale.y = height - 2 * CANVAS_PADDING;
    info.scene.offset.x = CANVAS_PADDING;
    info.scene.offset.y = height - CANVAS_PADDING;

    const pointBindFunction = (sprite: SpriteView, pred: IndexedScalars) => {
      const xValue = this.getValue(pred.id, key, model, label);
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
      sprite.SizePixel = !isSpecial ? SPRITE_SIZE_SM :
                          isSelected ? SPRITE_SIZE_MD : SPRITE_SIZE_LG;
    };

    info.points = info.scene.createSelection<IndexedScalars>()
        .onExit((sprite: SpriteView) => {sprite.SizePixel = 0;})
        .onInit(pointBindFunction)
        .onEnter(pointBindFunction)
        .onUpdate(pointBindFunction);
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

      const {brush, scene, model, key, points, xScale, yScale} = info;
      if (brush == null || scene == null || key == null || points == null ||
          model == null || xScale == null || yScale == null) continue;

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
      }

      brush.extent([[xMin, 0], [xMax, yMax + X_LABELS_PADDING]]);

      // Finally, update the Megaplot scene
      const {x, y} = scene.offset;
      scene.resize(scene.offset);
      scene.scale.x = xMax;
      scene.offset.x = x;
      scene.offset.y = y;
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

    const datasetScalars = this.datasetScalarKeys.map(
        field => this.renderPlot(field));

    const modelScalars: TemplateResult[] = [];
    for (const model of this.appState.currentModels) {
      const {output} = this.appState.getModelSpec(model);
      for (const [fieldName, fieldSpec] of Object.entries(output)) {
        if (fieldSpec instanceof MulticlassPreds) {
          const multiclassTemplates =
              this.renderClassificationGroup(fieldName, model);
          if (Array.isArray(multiclassTemplates)) {
            modelScalars.push(...multiclassTemplates);
          } else if (multiclassTemplates != null) {
            modelScalars.push(multiclassTemplates);
          }
        } else if (fieldSpec instanceof RegressionScore) {
          modelScalars.push(this.renderPlot(fieldName, model));
          const errorKey = `${fieldName}:${CalculatedColumnType.ERROR}`;
          modelScalars.push(this.renderPlot(errorKey, model));
          const sqerrKey = `${fieldName}:${CalculatedColumnType.SQUARED_ERROR}`;
          modelScalars.push(this.renderPlot(sqerrKey, model));
        } else if (fieldSpec instanceof Scalar) {
          modelScalars.push(this.renderPlot(fieldName, model));
        }
      }
    }

    // clang-format off
    return html`<div class="module-container">
      <div class="module-results-area">
        <div id='container'>
          ${datasetScalars}
          ${modelScalars}
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

  private renderClassificationGroup(
      key: string, model: string): TemplateResult | TemplateResult[] | null {
    const spec = this.appState.getModelSpec(model);
    const fieldSpec = spec.output[key];
    if (!(fieldSpec instanceof MulticlassPreds)) return null;
    const {vocab, null_idx} = fieldSpec;
    const controls = null_idx != null ? this.renderMarginSlider(model, key) :
                                        undefined;

    if (vocab?.length === 2 && null_idx != null) {
      // In the binary classification case, only render one plot that
      // displays the positive class.
      return this.renderPlot(key, model, vocab[1 - null_idx], controls);
    } else {
      // Otherwise, return one plot per label in the multiclass case.
      return vocab.map(label => this.renderPlot(key, model, label, controls));
    }
  }

  private renderPlot(key: string, model?: string, label?: string,
                     panelControls?: TemplateResult) {
    this.numPlotsRendered += 1;
    const modelComponent = model != null ? `${model}:` : '';
    const labelComponent = label != null ? `:${label}` : '';
    const id = `${modelComponent}${key}${labelComponent}`;
    const {primarySelectedId} = this.selectionService;
    const primaryPreds = this.preds.find(pred => pred.id === primarySelectedId);

    let plotLabel = `${id}`;
    if (primaryPreds != null) {
      const primaryValue =
          this.getValue(primaryPreds.id, key, model, label);
          plotLabel += ` - Value: ${formatForDisplay(primaryValue)}`;
    }

    if (!this.plots.has(id)) {
      const hidden = this.numPlotsRendered > MAX_DEFAULT_PLOTS;
      this.plots.set(id, {model, key, label, hidden});
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
        ${panelControls}
        <div  class="scatterplot" data-id=${id} @mousemove=${hover}
              @click=${select}>
          <div class="scene"></div>
          <div class="axes"></div>
        </div>
      </expansion-panel>
    </div>`;
    // clang-format on
  }

  private renderMarginSlider(model: string, key: string): TemplateResult {
    const margin = this.classificationService.getMargin(model, key);
    const callback = (e: CustomEvent<ThresholdChange>) => {
      this.classificationService.setMargin(model, key, e.detail.margin);
    };
    return html`<threshold-slider slot="bar-content" ?isThreshold=${false}
      .margin=${margin} ?showControls=${true} @threshold-changed=${callback}>
    </threshold-slider>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'scalar-module': ScalarModule;
  }
}
