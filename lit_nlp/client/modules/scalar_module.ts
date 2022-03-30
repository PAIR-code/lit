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
import {customElement} from 'lit/decorators';
import { html, svg} from 'lit';
import {computed, observable} from 'mobx';
// tslint:disable-next-line:ban-module-namespace-object-escape
const seedrandom = require('seedrandom');  // from //third_party/javascript/typings/seedrandom:bundle

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {ThresholdChange} from '../elements/threshold_slider';
import {D3Selection, formatForDisplay, IndexedInput, ModelInfoMap, ModelSpec, Preds, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, getThresholdFromMargin, isLitSubtype} from '../lib/utils';
import {FocusData} from '../services/focus_service';
import {ClassificationService, ColorService, DataService, GroupService, FocusService, RegressionService, SelectionService} from '../services/services';

import {styles} from './scalar_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

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
  private readonly regressionService = app.getService(RegressionService);
  private readonly focusService = app.getService(FocusService);
  private readonly dataService = app.getService(DataService);
  private readonly pinnedSelectionService =
      app.getService(SelectionService, 'pinned');

  private readonly inputIDToIndex = new Map();
  private resizeObserver!: ResizeObserver;
  private numPlotsRendered: number = 0;

  // Stores a BrushObject for each scatterplot that's drawn (in the case that
  // there are multiple pred keys).
  private readonly brushObjects: BrushObject[] = [];

  @observable private readonly isPlotHidden = new Map();
  @observable private preds: Preds[] = [];
  @observable private plotWidth = ScalarModule.maxPlotWidth;
  @observable private plotHeight = ScalarModule.minPlotHeight;
  private readonly plotTranslation: string =
      `translate(${ScalarModule.plotLeftMargin},${ScalarModule.plotTopMargin})`;

  @computed
  private get scalarKeys() {
    return this.groupService.numericalFeatureNames;
  }

  @computed
  private get scalarModelOutputKeys() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    return findSpecKeys(outputSpec, 'Scalar');
  }

  @computed
  private get classificationKeys() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    return findSpecKeys(outputSpec, 'MulticlassPreds');
  }

  override firstUpdated() {
    const modelSpec = this.appState.getModelSpec(this.model);
    this.classificationKeys.forEach((predKey) => {
      const predSpec = modelSpec.output[predKey];
      if (predSpec.null_idx != null && predSpec.vocab != null) {
        this.classificationService.setMargin(this.model, predKey, 0);
      }
    });

    this.makePlot();

    const getCurrentInputData = () => this.appState.currentInputData;
    this.reactImmediately(getCurrentInputData, currentInputData => {
      if (currentInputData != null) {
        // Get predictions from the backend for all input data and
        // display them by prediction score in the plot.
        this.updatePredictions(currentInputData);
      }
    });

    // Update predictions when new scalar columns exist to plot.
    const getScalarKeys = () => this.scalarKeys;
    this.reactImmediately(getScalarKeys, scalarKeys => {
      this.updatePredictions(this.appState.currentInputData);
    });

    // Update predictions when new data values are set.
    const getDataVals = () => this.dataService.dataVals;
    this.reactImmediately(getDataVals, dataVals => {
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
    this.reactImmediately(getMarginSettings, margins => {
      this.updateThreshold();
      // Update colors of datapoints as they may be based on predicted label.
      this.updateColors();
    });
    const getCompareEnabled = () => [
      this.appState.compareExamplesEnabled,
      this.pinnedSelectionService.primarySelectedInputData];
    this.reactImmediately(getCompareEnabled, () => {
      let referenceInputData = null;
      if (this.appState.compareExamplesEnabled) {
        referenceInputData =
            this.pinnedSelectionService.primarySelectedInputData;
      }
      this.updateReferenceSelection(referenceInputData);
    });

    const getSelectedColorOption = () => this.colorService.selectedColorOption;
    this.reactImmediately(getSelectedColorOption, selectedColorOption => {
      this.updateColors();
    });

    this.react(() => this.focusService.focusData, focusData => {
      this.updateHoveredPoint(focusData);
    });

    const container = this.shadowRoot!.getElementById('container')!;
    this.resizeObserver = new ResizeObserver(() => {
      this.resize();
    });
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
    const scatterplotClass = this.shadowRoot!.querySelectorAll('.scatterplot');

    for (const item of Array.from(scatterplotClass)) {
      const scatterplot = item as SVGGElement;
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
    const referencesInputData: IndexedInput[] = [];
    if (referenceInputData !== null) {
      referencesInputData.push(referenceInputData);
    }
    this.updateCallouts(
        referencesInputData, '#referenceOverlay', 'reference-point', true);
  }

  /**
   * Clears the previous callouts on the scatterplot and highlights new
   * called-out point(s). Callouts refers to points with special rendering,
   * such as selected or hovered points.
   */
  private updateCallouts(
      inputData: IndexedInput[], groupID: string, styleClass: string,
      detectHover: boolean) {
    const scatterplotClass = this.shadowRoot!.querySelectorAll('.scatterplot');

    for (const item of Array.from(scatterplotClass)) {
      const scatterplot = item as SVGGElement;

      const circles =
          d3.select(scatterplot).select('#dataPoints').selectAll('circle');

      const providedIndices =
          inputData.map((input) => this.inputIDToIndex.get(input.id));

      const providedDatapoints: string[][] = [];

      circles.each((d, i, e) => {
        if (providedIndices.includes(i)) {
          const x = d3.select(e[i]).attr('cx');
          const y = d3.select(e[i]).attr('cy');
          const color = d3.select(e[i]).style('fill');
          const id = d3.select(e[i]).attr('data-id');

          providedDatapoints.push([x, y, color, id]);
        }
      });

      // Select and clear overlay.
      const overlay = d3.select(scatterplot).select(groupID);
      overlay.selectAll('circle').remove();

      // Add provided datapoints.
      const overlayCircles = overlay.selectAll('circle')
                                 .data(providedDatapoints)
                                 .enter()
                                 .append('circle');
      overlayCircles
          .attr('cx', d => +d[0])
          .attr('cy', d => +d[1])
          .classed(styleClass, true)
          .style('fill', d => d[2]);

      // Add the appropriate event listeners.
      if (detectHover) {
        overlayCircles.on('mouseenter', (d, i, e) => {
          this.focusService.setFocusedDatapoint(d[3]);
        });
      } else {
        overlayCircles.on('click', (d, i, e) => {
          this.selectionService.setPrimarySelection(d[3]);
        });
      }
    }
  }

  /**
   * Get predictions from the backend for all input data and display by
   * prediction score in the plot.
   */
  private async updatePredictions(currentInputData: IndexedInput[]) {
    if (currentInputData == null) {
      return;
    }

    // TODO(lit-dev): consolidate to a single call here, with client-side cache.
    const dataset = this.appState.currentDataset;
    const promise = Promise.all([
      this.classificationService.getClassificationPreds(
          currentInputData, this.model, dataset),
      this.regressionService.getRegressionPreds(
          currentInputData, this.model, dataset),
      this.apiService.getPreds(
          currentInputData, this.model, dataset, ['Scalar']),
    ]);
    const results = await this.loadLatest('predictionScores', promise);
    if (results === null) {
      return;
    }
    const classificationPreds = results[0];
    const regressionPreds = results[1];
    const scalarPreds = results[2];
    if (classificationPreds == null && regressionPreds == null &&
        scalarPreds == null) {
      return;
    }

    const preds: Preds[] = [];
    for (let i = 0; i < classificationPreds.length; i++) {
      const currId = currentInputData[i].id;
      // TODO(lit-dev): structure this as a proper IndexedInput,
      // rather than having 'id' as a regular field.
      const pred = Object.assign(
          {}, classificationPreds[i], scalarPreds[i], regressionPreds[i],
          {id: currId});
      for (const scalarKey of this.scalarKeys) {
        pred[scalarKey] = this.dataService.getVal(currId, scalarKey);
      }
      preds.push(pred);
    }

    // Add the error info for any regression keys.
    if (regressionPreds != null && regressionPreds.length) {
      const ids = currentInputData.map(data => data.id);
      const regressionKeys = Object.keys(regressionPreds[0]);
      for (let j = 0; j < regressionKeys.length; j++) {
        const regressionInfo = await this.regressionService.getResults(
            ids, this.model, regressionKeys[j]);
        for (let i = 0; i < preds.length; i++) {
          preds[i][this.regressionService.getErrorKey(regressionKeys[j])] =
              regressionInfo[i].error;
        }
      }
    }

    this.preds = preds;
    this.updatePlotData();
  }

  /**
   * Returns the scale function for the scatter plot's x axis, for a given
   * key. If the key is for regression, we set the score range to be between the
   * min and max values of the regression scores.
   */
  private getXScale(key: string) {
    let scoreRange = [0, 1];

    const outputSpec = this.appState.currentModelSpecs[this.model]?.spec.output;
    if (outputSpec != null && isLitSubtype(outputSpec[key], 'Scalar')) {
      const scalarValues = this.preds.map((pred) => pred[key]);
      scoreRange = [Math.min(...scalarValues), Math.max(...scalarValues)];
      // If the range is 0 (all values are identical, then artificially increase
      // the range so that an X-axis is properly displayed.
      if (scoreRange[0] === scoreRange[1]) {
        scoreRange[0] = scoreRange[0] - .1;
        scoreRange[1] = scoreRange[1] + .1;
      }
    } else if (this.scalarKeys.indexOf(key) !== -1) {
      scoreRange = this.groupService.numericalFeatureRanges[key];
    }

    return d3.scaleLinear().domain(scoreRange).range([
      0, this.plotWidth - ScalarModule.plotLeftMargin * 1.5
    ]);
  }

  /**
   * Returns the scale function for the scatter plot's y axis.
   */
  private getYScale(key: string) {
    let scoreRange = [0, 1];

    const outputSpec = this.appState.currentModelSpecs[this.model]?.spec.output;
    if (outputSpec != null && isLitSubtype(outputSpec[key], 'Scalar')) {
      const ranges = this.regressionService.ranges[`${this.model}:${key}`];
      if (ranges != null && !isNaN(ranges.error[0]) &&
          !isNaN(ranges.error[1])) {
        scoreRange = ranges.error;
      }
    }
    return d3.scaleLinear().domain(scoreRange).range([
      this.plotHeight - ScalarModule.plotBottomMargin -
          ScalarModule.plotTopMargin,
      ScalarModule.plotTopMargin
    ]);
  }

  private getValue(preds: Preds, spec: ModelSpec, key: string, label: string) {
    // If for a multiclass prediction, return the top label score.
    if (isLitSubtype(spec.output[key], 'MulticlassPreds')) {
      const predictionLabels = spec.output[key].vocab!;
      const index = predictionLabels.indexOf(label);
      return preds[key][index];
    }
    // Otherwise, return the raw value.
    return preds[key];
  }

  /**
   * Re-renders threshold bar at the new threshold value and updates datapoint
   * colors.
   */
  private updateThreshold() {
    // Get classification threshold.
    const margins = this.classificationService.marginSettings[this.model] || {};

    const scatterplotClass = this.shadowRoot!.querySelectorAll('.scatterplot');

    for (const item of Array.from(scatterplotClass)) {
      const scatterplot = item as SVGGElement;
      const key = (item as HTMLElement).dataset['key'];

      if (key == null || this.scalarKeys.indexOf(key) !== -1) {
        return;
      }

      const spec = this.appState.getModelSpec(this.model);
      const labelList = spec.output[key].vocab!;

      // Don't render the threshold for regression or multiclass models.
      if (isLitSubtype(spec.output[key], 'Scalar') || labelList.length !== 2) {
        continue;
      }

      // If there is no margin set for the entire dataset (empty string) facet
      // name, then do not draw a threshold on the plot.
      if (margins[key] == null || margins[key][''] == null) {
        continue;
      }
      const threshold = getThresholdFromMargin(margins[key][""].margin);

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
    if (!container.offsetHeight || !container.offsetWidth) {
      return;
    }

    const scatterplots =
        Array.from(this.shadowRoot!.querySelectorAll('.scatterplot'));
    if (!scatterplots.length) {
      return;
    }
    this.plotWidth =
        container.offsetWidth - ScalarModule.plotLeftMargin * 1.5;

    this.plotHeight = 100;

    for (const item of Array.from(scatterplots)) {
      const key = (item as HTMLElement).dataset['key'];
      if (key == null) {
        continue;
      }

      const scatterplot = item as SVGGElement;
      const selected = d3.select(scatterplot)
                           .attr('width', this.plotWidth)
                           .attr('height', this.plotHeight);
      selected.selectAll('*').remove();

      // Add group for threshold marker.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'threshold')
          .attr('transform', this.plotTranslation);
      this.updateThreshold();

      // Add group for data points.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'dataPoints')
          .attr('transform', this.plotTranslation);

      // Add group for overlaying selected points.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'overlay')
          .attr('transform', this.plotTranslation);

      // Add group for overlaying primary selected points.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'primaryOverlay')
          .attr('transform', this.plotTranslation);

      // Add group for overlaying reference selected point.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'referenceOverlay')
          .attr('transform', this.plotTranslation);

      // Add group for overlaying hovered points.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'hoverOverlay')
          .attr('transform', this.plotTranslation);

      this.updateAllScatterplotColors(scatterplot);
    }
  }

  /**
   * Sets up brush and updates datapoints to reflect the current scalars.
   */
  private updatePlotData() {
    const scatterplots = this.shadowRoot!.querySelectorAll('.scatterplot');

    for (const item of Array.from(scatterplots)) {
      const key = (item as HTMLElement).dataset['key'];
      const label = (item as HTMLElement).dataset['label'];
      if (key == null || label == null) {
        return;
      }

      const spec = this.appState.getModelSpec(this.model);
      const isScalarKey = isLitSubtype(spec.output[key], 'Scalar');
      const regressionRanges =
          this.regressionService.ranges[`${this.model}:${key}`];
      const hasRegressionGroundTruth = regressionRanges != null &&
          !isNaN(regressionRanges.error[0]) &&
          !isNaN(regressionRanges.error[1]);

      const scatterplot = item as SVGGElement;
      const selected = d3.select(scatterplot);

      // Remove axes and brush groups before rerendering with current data.
      d3.select(scatterplot).select('#xAxis').remove();
      d3.select(scatterplot).select('#yAxis').remove();
      d3.select(scatterplot).select('#brushGroup').remove();

      const xScale = this.getXScale(key);
      const yScale = this.getYScale(key);

      const yAxisHeight = this.plotHeight - ScalarModule.plotBottomMargin;
      const xAxisTranslation =
          `translate(${ScalarModule.plotLeftMargin},${yAxisHeight})`;

      // Create axes.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'xAxis')
          .attr('transform', xAxisTranslation)
          .call(d3.axisBottom(xScale));

      const axisGenerator = d3.axisLeft(yScale);
      let numTicks = 0;
      if (isScalarKey && hasRegressionGroundTruth) {
        // Only display ticks if we have a meaningful y axis.
        numTicks = 5;
      }
      axisGenerator.ticks(numTicks);

      d3.select(scatterplot)
          .append('g')
          .attr('id', 'yAxis')
          .attr('transform', this.plotTranslation)
          .call(axisGenerator);

      // Create y axis label for regression models, where error is on the y
      // axis.
      if (isScalarKey && hasRegressionGroundTruth) {
        const errorTextY = ScalarModule.plotTopMargin +
                           this.plotHeight / 2 +
                           ScalarModule.yLabelOffsetY;

        d3.select(scatterplot)
          .append('text')
          .attr('transform', `translate(0,${errorTextY}) rotate(270)`)
          .style('text-anchor', 'middle')
          .text('error');
      }

      // Create zero line if this is a regression plot centered around zero.
      if (isScalarKey && hasRegressionGroundTruth) {
        const ranges = this.regressionService.ranges[`${this.model}:${key}`];
        const yRange = ranges.error;
        if (yRange[0] < 0 && yRange[1] > 0) {
          const halfHeight =
                (this.plotHeight - ScalarModule.plotBottomMargin) / 2 +
                ScalarModule.plotTopMargin;
          d3.select(scatterplot)
              .append('line')
              .attr('x1', ScalarModule.plotLeftMargin)
              .attr('y1', halfHeight)
              .attr('x2', xScale.range()[1] + ScalarModule.plotLeftMargin)
              .attr('y2', halfHeight)
              .style('stroke', ScalarModule.zeroLineColor);
        }
      }

      // Create a 2D brush for regression keys (since both axes are meaningful)
      // and a 1D brush for multiclass preds keys.
      const newBrush =
          (isScalarKey && hasRegressionGroundTruth) ? d3.brush() : d3.brushX();
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
      this.brushObjects.push({
        brush: newBrush,
        selection: brushGroup,
      });

      const dataPoints = d3.select(scatterplot).select('#dataPoints');
      dataPoints.selectAll('circle').remove();

      // Raise point groups to be in front of the d3.brush.
      dataPoints.raise();
      d3.select(scatterplot).select('#overlay').raise();
      d3.select(scatterplot).select('#referenceOverlay').raise();
      d3.select(scatterplot).select('#primaryOverlay').raise();
      d3.select(scatterplot).select('#hoverOverlay').raise();

      // Create scatterplot circles.
      const circles = dataPoints.selectAll('circle')
                          .data(this.preds)
                          .enter()
                          .append('circle');

      const rngSeed = 'lit';
      // tslint:disable-next-line:no-any ban-module-namespace-object-escape
      const rng = seedrandom(rngSeed);

      circles
          .attr('cx', d => xScale(this.getValue(d, spec, key, label)))
          .attr('cy', d => {
            if (isLitSubtype(spec.output[key], 'Scalar') &&
                hasRegressionGroundTruth) {
              return yScale(d[this.regressionService.getErrorKey(key)]);
            }
            // Otherwise, return a random value.
            return yScale(rng());
          })
          .style('fill', d => {
            const indexedInput = this.appState.getCurrentInputDataById(d['id']);
            return this.colorService.getDatapointColor(indexedInput);
          })
          .classed('point', true)
          .on('mouseenter', d => {
            this.focusService.setFocusedDatapoint(d['id']);
          })
          .each((d, i, e) => {
            this.inputIDToIndex.set(d['id'], i);
            // Store the inputID as a data attribute on the circle.
            d3.select(e[i]).attr('data-id', d['id']);
          });
    }
    this.updateGeneralSelection(this.selectionService.selectedInputData);
    this.updatePrimarySelection(this.selectionService.primarySelectedInputData);
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
        ${this.scalarModelOutputKeys.map(key => this.renderPlot(key, ''))}
        ${this.classificationKeys.map(key =>
          this.renderClassificationGroup(key))}
        ${this.scalarKeys.map(key => this.renderPlot(key, ''))}
      </div>
    `;
    // clang-format on
  }

  renderClassificationGroup(key: string) {
    const spec = this.appState.getModelSpec(this.model);
    const predictionLabels = spec.output[key].vocab!;

    // In the binary classification case, only render one plot that
    // displays the positive class.
    const nullIdx = spec.output[key].null_idx;
    if (predictionLabels.length === 2 && nullIdx != null) {
      return html`${this.renderPlot(key, predictionLabels[1 - nullIdx])}`;
    }

    // Otherwise, return one plot per label in the multiclass case.
    // clang-format off
    return html`
        ${(predictionLabels != null && nullIdx != null) ?
          this.renderMarginSlider(key) : null}
        ${predictionLabels.map(label => this.renderPlot(key, label))}`;
    // clang-format on
  }

  renderPlot(key: string, label: string) {
    const collapseByDefault = (this.numPlotsRendered > (MAX_DEFAULT_PLOTS - 1));
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

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, ['Scalar', 'MulticlassPreds']);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'scalar-module': ScalarModule;
  }
}
