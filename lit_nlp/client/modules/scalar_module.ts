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
import {html, svg} from 'lit';
import {customElement} from 'lit/decorators';
import {computed, observable} from 'mobx';
// tslint:disable-next-line:ban-module-namespace-object-escape
const seedrandom = require('seedrandom');  // from //third_party/javascript/typings/seedrandom:bundle

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {ThresholdChange} from '../elements/threshold_slider';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {D3Selection, formatForDisplay, IndexedInput, ModelInfoMap, ModelSpec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, getThresholdFromMargin, isLitSubtype} from '../lib/utils';
import {CalculatedColumnType, CLASSIFICATION_SOURCE_PREFIX, REGRESSION_SOURCE_PREFIX, SCALAR_SOURCE_PREFIX} from '../services/data_service';
import {FocusData} from '../services/focus_service';
import {ClassificationService, ColorService, DataService, FocusService, GroupService, SelectionService} from '../services/services';

import {styles} from './scalar_module.css';

/**
 * The maximum number of scatterplots to render on page load.
 */
export const MAX_DEFAULT_PLOTS = 2;

/**
 * Stores the d3 brush and the selected group that the brush is called on; used
 * to clear brush selections when the selected data changes.
 */
interface BrushObject {
  // tslint:disable-next-line:no-any
  brush: d3.BrushBehavior<any>;
  selection: D3Selection;
}

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
}

/**
 * A LIT module that visualizes prediction scores and other scalar values.
 */
@customElement('scalar-module')
export class ScalarModule extends LitModule {
  static override title = 'Scalars';
  static override numCols = 4;
  static override template = (model = '') => {
    return html`
      <scalar-module model=${model}>
      </scalar-module>`;
  };
  static maxPlotWidth = 900;
  static minPlotHeight = 100;
  static maxPlotHeight = 220;  // too sparse if taller than this
  static plotTopMargin = 6;
  static plotBottomMargin = 20;
  static plotLeftMargin = 32;
  static xLabelOffsetY = 30;
  static yLabelOffsetY = -25;
  static zeroLineColor = '#cccccc';

  static override get styles() {
    return [
      sharedStyles,
      styles,
    ];
  }

  private readonly colorService = app.getService(ColorService);
  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly groupService = app.getService(GroupService);
  private readonly focusService = app.getService(FocusService);
  private readonly dataService = app.getService(DataService);
  private readonly pinnedSelectionService =
      app.getService(SelectionService, 'pinned');

  private readonly inputIDToIndex = new Map();
  private readonly resizeObserver = new ResizeObserver(() => {this.resize();});
  private numPlotsRendered: number = 0;

  // Stores a BrushObject for each scatterplot that's drawn (in the case that
  // there are multiple pred keys).
  private readonly brushObjects: BrushObject[] = [];

  @observable private readonly isPlotHidden = new Map();
  @observable private preds: IndexedScalars[] = [];
  @observable private plotWidth = ScalarModule.maxPlotWidth;
  @observable private plotHeight = ScalarModule.minPlotHeight;
  private readonly plotTranslation: string =
      `translate(${ScalarModule.plotLeftMargin},${ScalarModule.plotTopMargin})`;

  @computed
  private get scalarColumnsToPlot() {
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

  @computed
  private get classificationKeys() {
    const {output} = this.appState.getModelSpec(this.model);
    return findSpecKeys(output, 'MulticlassPreds');
  }

  override firstUpdated() {
    const getDataChanges = () => [
      this.appState.currentInputData,
      this.dataService.dataVals,
      this.scalarColumnsToPlot
    ];
    this.reactImmediately(getDataChanges, () => {
      this.updatePredictions(this.appState.currentInputData);
    });

    const getSelectedInputData = () => this.selectionService.selectedInputData;
    this.reactImmediately(getSelectedInputData, selectedInputData => {
      if (selectedInputData != null) {
        // Clears the brush selection on all scatterplots.
        this.clearBrushes();
        // Clears previous selection and highlights the selected points.
        this.updateGeneralSelection(selectedInputData);
      }
    });

    const getPrimarySelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(
        getPrimarySelectedInputData, primarySelectedInputData => {
          // Clears previous primary selection and highlights the primary
          // selected point.
          this.updatePrimarySelection(primarySelectedInputData);
        });

    const getMarginSettings = () =>
        this.classificationService.allMarginSettings;
    this.reactImmediately(getMarginSettings, () => {
      this.updateThreshold();
      // Update colors of datapoints as they may be based on predicted label.
      this.updateColors();
    });

    const getCompareEnabled = () =>
        this.pinnedSelectionService.primarySelectedInputData;
    this.reactImmediately(getCompareEnabled, () => {
      const pinned = this.pinnedSelectionService.primarySelectedInputData;
      this.updateReferenceSelection(pinned);
    });

    const getSelectedColorOption = () => this.colorService.selectedColorOption;
    this.reactImmediately(getSelectedColorOption, () => {
      this.updateColors();
    });

    this.react(() => this.focusService.focusData, focusData => {
      this.updateHoveredPoint(focusData);
    });

    this.react(() => this.preds, () => {
      this.makePlot();
      this.updatePlotData();
    });

    const container = this.shadowRoot!.getElementById('container')!;
    this.resizeObserver.observe(container);
  }

  private resize() {
    this.makePlot();
    this.updatePlotData();
  }

  /*
   * Clears the brush selection on all scatterplots.
   */
  private clearBrushes() {
    for (const brushObject of this.brushObjects) {
      brushObject.selection.call(brushObject.brush.move, null);
    }
  }

  /**
   * Updates the colors of datapoints to match color service settings.
   */
  private updateColors() {
    const scatterplots =
        this.shadowRoot!.querySelectorAll<SVGSVGElement>('.scatterplot');

    for (const scatterplot of scatterplots) {
      this.updateAllScatterplotColors(scatterplot);
    }
  }

  /**
   * Updates the scatterplot with the new selection.
   */
  private updateGeneralSelection(selectedInputData: IndexedInput[]) {
    this.updateCallouts(selectedInputData, '#overlay', 'selected-point', true);
  }

  /**
   * Updates the scatterplot with the new primary selection.
   */
  private updatePrimarySelection(primarySelectedInputData: IndexedInput|null) {
    const selectedInputData: IndexedInput[] = [];
    if (primarySelectedInputData !== null) {
      selectedInputData.push(primarySelectedInputData);
    }
    this.updateCallouts(
        selectedInputData, '#primaryOverlay', 'primary-selected-point', true);
  }

  /**
   * Updates the scatterplot with the new hovered selection.
   */
  private updateHoveredPoint(focusData: FocusData|null) {
    let hoveredData: IndexedInput[] = [];
    if (focusData !== null) {
      hoveredData = this.appState.getExamplesById([focusData.datapointId]);
    }
    this.updateCallouts(
        hoveredData, '#hoverOverlay', 'hovered-point', false);
  }

  /**
   * Updates the scatterplot with the new reference selection.
   */
  private updateReferenceSelection(referenceInputData: IndexedInput|null) {
    const references: IndexedInput[] =
        referenceInputData ? [referenceInputData] : [];
    this.updateCallouts(
        references, '#referenceOverlay', 'reference-point', true);
  }

  /**
   * Clears the previous callouts on the scatterplot and highlights new
   * called-out point(s). Callouts refers to points with special rendering,
   * such as selected or hovered points.
   */
  private updateCallouts(
      inputData: IndexedInput[], groupID: string, styleClass: string,
      detectHover: boolean) {
    const providedIndices =
        inputData.map(input => this.inputIDToIndex.get(input.id));
    const scatterplotClass =
        this.shadowRoot!.querySelectorAll<SVGSVGElement>('.scatterplot');

    for (const scatterplot of scatterplotClass) {
      const circles =
          d3.select(scatterplot).select('#dataPoints').selectAll('circle');
      const providedDatapoints: string[][] = [];

      circles.each((d, i, e) => {
        if (providedIndices.includes(i)) {
          const circle = d3.select(e[i]);
          const props = [
            circle.attr('cx'),
            circle.attr('cy'),
            circle.style('fill'),
            circle.attr('data-id')
          ];
          providedDatapoints.push(props);
        }
      });

      // Select and clear overlay.
      const overlay = d3.select(scatterplot).select(groupID);
      overlay.selectAll('circle').remove();

      // Add provided datapoints.
      const overlayCircles = overlay.selectAll('circle')
          .data(providedDatapoints)
          .enter()
          .append('circle')
          .attr('cx', d => Number(d[0]))
          .attr('cy', d => Number(d[1]))
          .attr('data-id', d => d[3])
          .classed(styleClass, true)
          .style('fill', d => d[2]);

      // Add the appropriate event listeners.
      if (detectHover) {
        overlayCircles.on('mouseenter', d => {
          this.focusService.setFocusedDatapoint(d[3]);
        });
      } else {
        overlayCircles.on('click', d => {
          this.selectionService.setPrimarySelection(d[3]);
        });
      }
    }
  }

  /**
   * Get predictions from the backend for all input data and display by
   * prediction score in the plot.
   */
  private async updatePredictions(currentInputData?: IndexedInput[]) {
    if (currentInputData == null) {return;}

    const preds: IndexedScalars[] = [];
    for (const {id} of currentInputData) {
      const pred: IndexedScalars = {id, data: {}};
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
    if (isLitSubtype(output[key], 'Scalar')) {
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

    return d3.scaleLinear().domain(scoreRange).range([
      0, this.plotWidth - ScalarModule.plotLeftMargin * 1.5
    ]);
  }

  /**
   * Returns the scale function for the scatter plot's y axis.
   */
  private getYScale(column: string) {
    const scale = d3.scaleLinear().domain([0, 1]).range([
      this.plotHeight - ScalarModule.plotBottomMargin -
          ScalarModule.plotTopMargin,
      ScalarModule.plotTopMargin
    ]);

    const colInfo = this.dataService.getColumnInfo(column);
    if (colInfo == null || column.includes(CalculatedColumnType.ERROR)) {
      return scale;  // Input field or a regression error field.
    }

    const errorName = this.dataService.getColumnName(
        this.model, colInfo.key, CalculatedColumnType.ERROR);
    const errFeatInfo = this.dataService.getColumnInfo(errorName);
    if (errFeatInfo == null) {return scale;}  // Non-regression output field

    const values = this.dataService.getColumn(errorName);
    const range = d3.extent(values);
    if (range != null && !range.some(isNaN)) {
      // Make the domain symmetric around 0
      const largest = Math.max(...(range as number[]).map(Math.abs));
      scale.domain([-largest, largest]);
    }

    return scale;   // Regression output field
  }

  private getValue(preds: IndexedScalars, spec: ModelSpec, key: string,
                   label: string): number | undefined {
    // If for a multiclass prediction and the DataService has loaded the
    // classification results, return the label score from the array.
    if (isLitSubtype(spec.output[key], 'MulticlassPreds')) {
      const {vocab} = spec.output[key];
      const index = vocab!.indexOf(label);
      const classPreds = preds.data[key];
      if (Array.isArray(classPreds)) return classPreds[index];
    }
    // Otherwise, return the value of data[key], which may be undefined if the
    // DataService async calls are still pending.
    return preds.data[key] as number | undefined;
  }

  /**
   * Re-renders threshold bar at the new threshold value and updates datapoint
   * colors.
   */
  private updateThreshold() {
    // Get classification threshold.
    const margins = this.classificationService.marginSettings[this.model] || {};

    const scatterplotClass =
        this.shadowRoot!.querySelectorAll<SVGSVGElement>('.scatterplot');

    for (const scatterplot of scatterplotClass) {
      const {'key': key} = scatterplot.dataset;
      if (key == null || !this.classificationKeys.includes(key)) continue;

      // Don't render the threshold for regression or multiclass models.
      const {output} = this.appState.getModelSpec(this.model);
      const {vocab} = output[key];
      if (isLitSubtype(output[key], 'Scalar') || vocab?.length !== 2) continue;

      // If there is no margin set for the entire dataset (empty string) facet
      // name, then do not draw a threshold on the plot.
      if (margins[key] == null || margins[key][''] == null) continue;

      const threshold = getThresholdFromMargin(margins[key][''].margin);
      const thresholdSelection = d3.select(scatterplot).select('#threshold');
      thresholdSelection.selectAll('line').remove();

      const x = this.getXScale(key)(threshold);
      const [minY, maxY] = this.getYScale(key).range();

      // Create threshold marker.
      thresholdSelection.append('line')
          .attr('x1', x)
          .attr('y1', minY)
          .attr('x2', x)
          .attr('y2', maxY)
          .style('stroke', 'black');
    }
  }

  /**
   * Updates datapoint colors on a given scatterplot element to the current
   * color service settings.
   */
  private updateAllScatterplotColors(scatterplot: SVGGElement) {
    this.updateScatterplotColors(scatterplot, '#dataPoints');
    this.updateScatterplotColors(scatterplot, '#overlay');
    this.updateScatterplotColors(scatterplot, '#referenceOverlay');
    this.updateScatterplotColors(scatterplot, '#primaryOverlay');
  }

  /**
   * Updates datapoint colors for a given scatterplot group to the current color
   * service settings.
   */
  private updateScatterplotColors(scatterplot: SVGGElement, groupID: string) {
    const dataPoints = d3.select(scatterplot).select(groupID);
    // Update point colors.
    dataPoints.selectAll('circle').style('fill', (d, i, e) => {
      // The #overlay and #primaryOverlay groups use a string[4] where the last
      // element is the id, everything else stores the id in the data-id
      const id = Array.isArray(d) ? d[d.length - 1] :
                                    d3.select(e[i]).attr('data-id');
      const indexedInput = this.appState.getCurrentInputDataById(id);
      return this.colorService.getDatapointColor(indexedInput);
    });
  }

  /**
   * Sets up the scatterplot svgs, axes, and point groups.
   */
  private makePlot() {
    const container = this.shadowRoot!.getElementById('container')!;
    if (!container.offsetHeight || !container.offsetWidth) {return;}

    const scatterplots =
        this.shadowRoot!.querySelectorAll<SVGSVGElement>('.scatterplot');
    if (scatterplots.length < 1) {return;}

    this.plotHeight = 100;
    this.plotWidth =
        container.offsetWidth - ScalarModule.plotLeftMargin * 1.5;

    for (const scatterplot of scatterplots) {
      const {'key': key} = scatterplot.dataset;
      if (key == null) {continue;}

      const selected = d3.select(scatterplot)
                         .attr('width', this.plotWidth)
                         .attr('height', this.plotHeight);
      selected.selectAll('*').remove();

      const ids = [
          'threshold', 'dataPoints', 'overlay', 'primaryOverlay',
          'referenceOverlay', 'hoverOverlay'];

      for (const id of ids) {
        selected.append('g')
            .attr('id', id)
            .attr('transform', this.plotTranslation);
      }

      this.updateAllScatterplotColors(scatterplot);
    }

    this.updateThreshold();
  }

  /**
   * Sets up brush and updates datapoints to reflect the current scalars.
   */
  private updatePlotData() {
    const scatterplots =
        this.shadowRoot!.querySelectorAll<SVGSVGElement>('.scatterplot');

    for (const scatterplot of scatterplots) {
      const {'key': key, 'label': label} = scatterplot.dataset;
      if (key == null || label == null) {return;}

      const spec = this.appState.getModelSpec(this.model);
      const errorFeatName = `${key}:${CalculatedColumnType.ERROR}`;
      const errFeatInfo = this.dataService.getColumnInfo(errorFeatName);
      const hasRegressionGroundTruth =
          errFeatInfo?.source.includes(REGRESSION_SOURCE_PREFIX) &&
          !key.includes(CalculatedColumnType.ERROR);
      const errFeatCol = this.dataService.getColumn(errorFeatName);
      const yRange = hasRegressionGroundTruth ? d3.extent(errFeatCol) : null;

      const selected = d3.select(scatterplot);

      // Remove axes and brush groups before rerendering with current data.
      selected.select('#xAxis').remove();
      selected.select('#yAxis').remove();
      selected.select('#brushGroup').remove();

      const xScale = this.getXScale(key);
      const yScale = this.getYScale(key);

      const yAxisHeight = this.plotHeight - ScalarModule.plotBottomMargin;
      const xAxisTranslation =
          `translate(${ScalarModule.plotLeftMargin},${yAxisHeight})`;

      // Create axes.
      selected.append('g')
          .attr('id', 'xAxis')
          .attr('transform', xAxisTranslation)
          .call(d3.axisBottom(xScale));

      const axisGenerator = d3.axisLeft(yScale);
      const numTicks = hasRegressionGroundTruth ? 5 : 0;
      axisGenerator.ticks(numTicks);

      selected.append('g')
          .attr('id', 'yAxis')
          .attr('transform', this.plotTranslation)
          .call(axisGenerator);

      // Add labels and lines for regression plots
      if (hasRegressionGroundTruth) {
        const errorTextY = ScalarModule.plotTopMargin + this.plotHeight / 2 +
                           ScalarModule.yLabelOffsetY;

        // Add axis label
        selected.append('text')
          .attr('transform', `translate(0,${errorTextY}) rotate(270)`)
          .style('text-anchor', 'middle')
          .text('error');

        // Create zero line at y = 0
        if (yRange != null && yRange[0] < 0 && yRange[1] > 0) {
          const halfHeight =
                (this.plotHeight - ScalarModule.plotBottomMargin) / 2 +
                ScalarModule.plotTopMargin;
          selected.append('line')
              .attr('x1', ScalarModule.plotLeftMargin)
              .attr('y1', halfHeight)
              .attr('x2', xScale.range()[1] + ScalarModule.plotLeftMargin)
              .attr('y2', halfHeight)
              .style('stroke', ScalarModule.zeroLineColor);
        }
      }

      // Create a 2D brush for regression keys (since both axes are meaningful)
      // and a 1D brush for multiclass preds keys.
      const newBrush = hasRegressionGroundTruth ? d3.brush() : d3.brushX();
      newBrush
          .extent([
            [0, ScalarModule.plotTopMargin],
            [
              this.plotWidth - ScalarModule.plotLeftMargin,
              this.plotHeight - ScalarModule.plotBottomMargin
            ]
          ])
          .on('start end', () => {
            if (!d3.event.sourceEvent) {
              return;
            }
            this.selectBrushedPoints(scatterplot);
          });

      const brushGroup = selected.append('g')
          .attr('id', 'brushGroup')
          .attr('transform', `translate(${ScalarModule.plotLeftMargin},0)`)
          .on('mouseenter', () => { this.focusService.clearFocus(); })
          .call(newBrush);

      // Store brush and selection group to be used for clearing the brush.
      this.brushObjects.push({brush: newBrush, selection: brushGroup});


      /**
       * Random number generator for jitter Y positions.
       *
       * Seeded random number generators produce the same value on sequential
       * calls to `.seedrandom()` across instances initialized with the same
       * seed, in this case the string `'lit'`. We use this to maintain stable Y
       * axis positions in and across jiter plots (i.e., plots of non-regression
       * scalars), when `.updatePlotData()` is called, e.g., after a resize, so
       * we cannot move this to a module constant.
       */
      // tslint:disable-next-line:no-any ban-module-namespace-object-escape
      const rng = seedrandom('lit');

      const dataPoints = selected.select('#dataPoints');
      dataPoints.selectAll('circle').remove();

      // Raise point groups to be in front of the d3.brush.
      dataPoints.raise();
      selected.select('#overlay').raise();
      selected.select('#referenceOverlay').raise();
      selected.select('#primaryOverlay').raise();
      selected.select('#hoverOverlay').raise();
      // Create scatterplot circles.
      dataPoints.selectAll('circle')
          .data(this.preds)
          .enter()
          .append('circle')
          .attr('cx', d => {
            // This value can be undefined if an async call originating in
            // DataService has not returned, so we put these points 8px off the
            // right side of the plot
            const rawVal = this.getValue(d, spec, key, label);
            return rawVal == null ? xScale.range()[1] + 8 : xScale(rawVal);
          })
          .attr('cy', d => {
            // If a regression use the error value, else use a random number
            const rawVal = hasRegressionGroundTruth ?
                this.dataService.getVal(d.id, errorFeatName) : rng();
            const scaledVal = yScale(rawVal);
            return scaledVal;
          })
          .attr('data-id', (d, i) => {
            // Store the inputID as a data attribute on the circle.
            this.inputIDToIndex.set(d.id, i);
            return d.id;
          })
          .style('fill', d => {
            const indexedInput = this.appState.getCurrentInputDataById(d.id);
            return this.colorService.getDatapointColor(indexedInput);
          })
          .classed('point', true)
          .on('mouseenter', d => {
            this.focusService.setFocusedDatapoint(d.id);
          });
    }

    this.updateGeneralSelection(this.selectionService.selectedInputData);
    this.updatePrimarySelection(
        this.selectionService.primarySelectedInputData);
    this.updateReferenceSelection(
        this.pinnedSelectionService.primarySelectedInputData);
  }

  private selectBrushedPoints(scatterplot: SVGGElement) {
    const bounds = d3.event.selection;
    if (bounds != null && bounds.length > 0) {
      const ids: string[] = [];
      const hasYDimension = (typeof bounds[0]) !== 'number';
      const boundsX = hasYDimension ? [bounds[0][0], bounds[1][0]] : bounds;
      const boundsY = hasYDimension ? [bounds[0][1], bounds[1][1]] : null;

      d3.select(scatterplot)
          .select('#dataPoints')
          .selectAll('circle')
          .each((d, i, e) => {
            const x = +d3.select(e[i]).attr('cx');
            if (x < boundsX[0] || x > boundsX[1]) {
              return;
            }
            if (hasYDimension && boundsY != null) {
              const y = +d3.select(e[i]).attr('cy');
              if (y < boundsY[0] || y > boundsY[1]) {
                return;
              }
            }
            const id = d3.select(e[i]).attr('data-id');
            ids.push(id);
          });
      this.selectionService.selectIds(ids);
    }
  }

  override render() {
    this.numPlotsRendered = 0;
    // clang-format off
    return html`
      <div id='container'>
        ${this.classificationKeys.map(key =>
          this.renderClassificationGroup(key))}
        ${this.scalarColumnsToPlot.map(key => this.renderPlot(key, ''))}
      </div>
    `;
    // clang-format on
  }

  renderClassificationGroup(key: string) {
    const spec = this.appState.getModelSpec(this.model);
    const {vocab, null_idx} = spec.output[key];
    if (vocab == null) return;

    // In the binary classification case, only render one plot that
    // displays the positive class.
    if (vocab?.length === 2 && null_idx != null) {
      return html`${this.renderPlot(key, vocab[1 - null_idx])}`;
    }

    // Otherwise, return one plot per label in the multiclass case.
    // clang-format off
    return html`
        ${null_idx != null ? this.renderMarginSlider(key) : null}
        ${vocab.map(label => this.renderPlot(key, label))}`;
    // clang-format on
  }

  renderPlot(key: string, label: string) {
    const collapseByDefault = this.numPlotsRendered >= MAX_DEFAULT_PLOTS;
    this.numPlotsRendered++;

    const axisTitle = label ? `${key}:${label}` : key;
    let selectedValue = '';
    if (this.selectionService.primarySelectedId != null) {
      const selectedIndex = this.appState.getIndexById(
          this.selectionService.primarySelectedId);
      if (selectedIndex != null && selectedIndex < this.preds.length &&
          this.preds[selectedIndex] != null) {
        const spec = this.appState.getModelSpec(this.model);
        const displayVal = formatForDisplay(
            this.getValue(this.preds[selectedIndex], spec, key, label));
        selectedValue = `Value: ${displayVal}`;
      }
    }

    const plotLabel =
        `${axisTitle}${selectedValue ? ` - ${selectedValue}` : ''}`;

    const toggleCollapse = () => {
      const isHidden = (this.isPlotHidden.get(axisTitle) == null) ?
          collapseByDefault : this.isPlotHidden.get(axisTitle);
      this.isPlotHidden.set(axisTitle, !isHidden);
    };
    // This plot's value in isPlotHidden gets set in toggleCollapse and is null
    // before the user opens/closes it for the first time. This uses the
    // collapseByDefault setting if isPlotHidden hasn't been set yet.
    const isHidden = (this.isPlotHidden.get(axisTitle) == null) ?
        collapseByDefault : this.isPlotHidden.get(axisTitle);

    // clang-format off
    return html`
        <div class='plot-holder'>
          <expansion-panel .label=${plotLabel} ?expanded=${!isHidden}
                            padLeft padRight
                            @expansion-toggle=${toggleCollapse}>
            ${isHidden ? null : html`
            <div class='scatterplot-background'>
              ${svg`<svg class='scatterplot' data-key='${key}'
                      data-label='${label}'>
                    </svg>`}
            </div>`}
          </expansion-panel>
        </div>`;
    // clang-format on
  }

  renderMarginSlider(key: string) {
    const margin = this.classificationService.getMargin(this.model, key);
    const callback = (e: CustomEvent<ThresholdChange>) => {
      this.classificationService.setMargin(
          this.model, key, e.detail.margin);
    };
    return html`<threshold-slider .margin=${margin} label=${key}
                  ?isThreshold=${false} ?showControls=${true}
                  @threshold-changed=${callback}>
                </threshold-slider>`;
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap) {
    return doesOutputSpecContain(modelSpecs, ['Scalar', 'MulticlassPreds']);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'scalar-module': ScalarModule;
  }
}
