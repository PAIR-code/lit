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

// tslint:disable:no-new-decorators
// taze: ResizeObserver from //third_party/javascript/typings/resize_observer_browser
import * as d3 from 'd3';
import {customElement, html, property, svg} from 'lit-element';
import {computed, observable} from 'mobx';
// tslint:disable-next-line:ban-module-namespace-object-escape
const seedrandom = require('seedrandom');  // from //third_party/javascript/typings/seedrandom:bundle

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {D3Selection, IndexedInput, ModelsMap, NumericSetting, Preds, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, getThresholdFromMargin, isLitSubtype} from '../lib/utils';
import {PredictionsService, ClassificationService, ColorService, RegressionService} from '../services/services';

import {styles} from './prediction_score_module.css';
import {styles as sharedStyles} from './shared_styles.css';

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
 * A LIT module that visualizes prediction scores.
 */
@customElement('prediction-score-module')
export class PredictionScoreModule extends LitModule {
  static title = 'Prediction Score';
  static numCols = 4;
  static template = (model = '') => {
    return html`
      <prediction-score-module model=${model}>
      </prediction-score-module>`;
  };
  static maxPlotWidth = 900;
  static minPlotHeight = 100;
  static maxPlotHeight = 250;  // too sparse if taller than this
  static plotBottomMargin = 35;
  static plotLeftMargin = 5;
  static xLabelOffsetY = 30;
  static yLabelOffsetX = -32;
  static yLabelOffsetY = -25;
  static zeroLineColor = '#cccccc';

  static get styles() {
    return [
      sharedStyles,
      styles,
    ];
  }

  private readonly colorService = app.getService(ColorService);
  private readonly predictionsService = app.getService(PredictionsService);
  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly regressionService = app.getService(RegressionService);

  private readonly inputIDToIndex = new Map();
  private resizeObserver!: ResizeObserver;

  // Stores a BrushObject for each scatterplot that's drawn (in the case that
  // there are multiple pred keys).
  private readonly brushObjects: BrushObject[] = [];

  @observable private preds: Preds[] = [];
  @observable private plotWidth = PredictionScoreModule.maxPlotWidth;
  @observable private plotHeight = PredictionScoreModule.minPlotHeight;

  @computed
  private get scalarKeys() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    return findSpecKeys(outputSpec, 'Scalar');
  }

  @computed
  private get classificationKeys() {
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    return findSpecKeys(outputSpec, 'MulticlassPreds');
  }

  firstUpdated() {
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
        this.updatePredictions();
      }
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

    const getSelectedColorOption = () => this.colorService.selectedColorOption;
    this.reactImmediately(getSelectedColorOption, selectedColorOption => {
      this.updateColors();
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
    this.updateSelection(selectedInputData, '#overlay', 'selected-point');
  }

  /**
   * Updates the scatterplot with the new primary selection.
   */
  private updatePrimarySelection(primarySelectedInputData: IndexedInput|null) {
    const selectedInputData: IndexedInput[] = [];
    if (primarySelectedInputData !== null) {
      selectedInputData.push(primarySelectedInputData);
    }
    this.updateSelection(
        selectedInputData, '#primaryOverlay', 'primary-selected-point');
  }

  /**
   * Clears the previous selection on the scatterplot and highlights new
   * selected point(s).
   */
  private updateSelection(
      selectedInputData: IndexedInput[], groupID: string, styleClass: string) {
    const scatterplotClass = this.shadowRoot!.querySelectorAll('.scatterplot');

    for (const item of Array.from(scatterplotClass)) {
      const scatterplot = item as SVGGElement;

      const circles =
          d3.select(scatterplot).select('#dataPoints').selectAll('circle');

      const selectedIndices =
          selectedInputData.map((input) => this.inputIDToIndex.get(input.id));

      const selectedDatapoints: string[][] = [];

      circles.each((d, i, e) => {
        if (selectedIndices.includes(i)) {
          const x = d3.select(e[i]).attr('cx');
          const y = d3.select(e[i]).attr('cy');
          const color = d3.select(e[i]).style('fill');
          const id = d3.select(e[i]).attr('data-id');

          selectedDatapoints.push([x, y, color, id]);
        }
      });

      // Select and clear overlay.
      const overlay = d3.select(scatterplot).select(groupID);
      overlay.selectAll('circle').remove();

      // Add selected datapoints.
      const overlayCircles = overlay.selectAll('circle')
                                 .data(selectedDatapoints)
                                 .enter()
                                 .append('circle');
      overlayCircles
          .attr(
              'cx',
              (d) => {
                return +d[0];
              })
          .attr(
              'cy',
              (d) => {
                return +d[1];
              })
          .classed(styleClass, true)
          .style(
              'fill',
              (d) => {
                return d[2];
              })
          .on('click', (d, i, e) => {
            this.selectionService.setPrimarySelection(d[3]);
          });
    }
  }

  /**
   * Get predictions from the backend for all input data and display by
   * prediction score in the plot.
   */
  private async updatePredictions() {
    const promise = this.predictionsService.ensurePredictionsFetched(this.model);
    const preds = await this.loadLatest('ensurePredictionsFetched', promise);
    if (preds) {
      this.preds = preds;
      this.updatePlotData();
    }
  }

  /**
   * Returns the scale function for the scatter plot's x axis, for a given
   * prediction key. If the key is for regression, we set the score
   * range to be between the min and max values of the regression score
   * predictions.
   */
  private getXScale(key: string) {
    let scoreRange = [0, 1];

    const outputSpec = this.appState.currentModelSpecs[this.model]?.spec.output;
    if (outputSpec != null && isLitSubtype(outputSpec[key], 'Scalar')) {
      const scalarValues = this.preds.map((pred) => pred[key]);
      scoreRange = [Math.min(...scalarValues), Math.max(...scalarValues)];
    }

    return d3.scaleLinear().domain(scoreRange).range([
      0, this.plotWidth - PredictionScoreModule.plotLeftMargin
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
      this.plotHeight - PredictionScoreModule.plotBottomMargin, 0
    ]);
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

      if (key == null) {
        return;
      }

      const spec = this.appState.getModelSpec(this.model);
      const labelList = spec.output[key].vocab!;

      // Don't render the threshold for regression or multiclass models.
      if (isLitSubtype(spec.output[key], 'Scalar') || labelList.length !== 2) {
        continue;
      }

      if (margins[key] == null) {
        continue;
      }
      const threshold = getThresholdFromMargin(+margins[key]);

      const thresholdSelection = d3.select(scatterplot).select('#threshold');
      thresholdSelection.selectAll('line').remove();

      const xScale = this.getXScale(key);
      const yScale = this.getYScale(key);

      // Create threshold marker.
      thresholdSelection.append('line')
          .attr('x1', xScale(threshold))
          .attr('y1', yScale(0))
          .attr('x2', xScale(threshold))
          .attr(
              'y2',
              yScale(this.plotHeight - PredictionScoreModule.plotBottomMargin))
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
  }

  /**
   * Updates datapoint colors for a given scatterplot group to the current color
   * service settings.
   */
  private updateScatterplotColors(scatterplot: SVGGElement, groupID: string) {
    const dataPoints = d3.select(scatterplot).select(groupID);
    // Update point colors.
    dataPoints.selectAll('circle').style('fill', (d, i, e) => {
      const id = d3.select(e[i]).attr('data-id');
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
        container.offsetWidth - PredictionScoreModule.plotLeftMargin * 2;

    // TODO(lit-dev): replace this with something that looks at the containing
    // div instead of estimating from the full module container.
    const perRowMargin = 15;  // should match .plot_holder margin-bottom in css
    const proposedPlotHeight =
        (container.offsetHeight - PredictionScoreModule.plotBottomMargin) /
            scatterplots.length -
        perRowMargin;
    this.plotHeight = Math.min(
        Math.max(PredictionScoreModule.minPlotHeight, proposedPlotHeight),
        PredictionScoreModule.maxPlotHeight);

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
          .attr(
              'transform',
              'translate(' + PredictionScoreModule.plotLeftMargin.toString() +
                  ',0)');
      this.updateThreshold();

      // Add group for data points.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'dataPoints')
          .attr(
              'transform',
              'translate(' + PredictionScoreModule.plotLeftMargin.toString() +
                  ',0)');

      // Add group for overlaying selected points.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'overlay')
          .attr(
              'transform',
              'translate(' + PredictionScoreModule.plotLeftMargin.toString() +
                  ',0)');

      // Add group for overlaying primary selected points.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'primaryOverlay')
          .attr(
              'transform',
              'translate(' + PredictionScoreModule.plotLeftMargin.toString() +
                  ',0)');

      this.updateAllScatterplotColors(scatterplot);
    }
  }

  /**
   * Sets up brush and updates datapoints to reflect the current predictions.
   */
  private updatePlotData() {
    const scatterplots = this.shadowRoot!.querySelectorAll('.scatterplot');

    for (const item of Array.from(scatterplots)) {
      const key = (item as HTMLElement).dataset['key'];
      const label = (item as HTMLElement).dataset['label'];
      if (key == null || label == null) {
        return;
      }
      const axisTitle = (item as HTMLElement).dataset['axisTitle'] ?? key;

      const spec = this.appState.getModelSpec(this.model);
      const isScalarKey = isLitSubtype(spec.output[key], 'Scalar');
      const regressionRanges =
          this.regressionService.ranges[`${this.model}:${key}`];
      const hasRegressionGroundTruth = regressionRanges != null &&
          !isNaN(regressionRanges.error[0]) &&
          !isNaN(regressionRanges.error[1]);
      const isMulticlassPredsKey =
          isLitSubtype(spec.output[key], 'MulticlassPreds');

      const scatterplot = item as SVGGElement;
      const selected = d3.select(scatterplot);

      // Remove axes and brush groups before rerendering with current data.
      d3.select(scatterplot).select('#xAxis').remove();
      d3.select(scatterplot).select('#yAxis').remove();
      d3.select(scatterplot).select('#brushGroup').remove();

      const xScale = this.getXScale(key);
      const yScale = this.getYScale(key);

      // Create axes.
      d3.select(scatterplot)
          .append('g')
          .attr('id', 'xAxis')
          .attr(
              'transform',
              'translate(' + PredictionScoreModule.plotLeftMargin.toString() +
                  ',' +
                  (this.plotHeight - PredictionScoreModule.plotBottomMargin)
                      .toString() +
                  ')')
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
          .attr(
              'transform',
              'translate(' + PredictionScoreModule.plotLeftMargin.toString() +
                  ', 0)')
          .call(axisGenerator);

      // Create x axis label.
      d3.select(scatterplot)
          .append('text')
          .attr(
              'transform',
              'translate(' + (this.plotWidth / 2).toString() + ' ,' +
                  (this.plotHeight - PredictionScoreModule.plotBottomMargin +
                   PredictionScoreModule.xLabelOffsetY)
                      .toString() +
                  ')')
          .style('text-anchor', 'middle')
          .text(axisTitle);

      // Create y axis label for regression models, where error is on the y
      // axis.
      if (isScalarKey && hasRegressionGroundTruth) {
        d3.select(scatterplot)
            .append('text')
            .attr(
                'transform',
                'translate(' +
                    (PredictionScoreModule.plotLeftMargin +
                     PredictionScoreModule.yLabelOffsetX)
                        .toString() +
                    ', ' +
                    (this.plotHeight / 2 + PredictionScoreModule.yLabelOffsetY)
                        .toString() +
                    ') rotate(270)')
            .style('text-anchor', 'middle')
            .text('error');
      }

      // Create zero line if this is a regression plot centered around zero.
      if (isScalarKey && hasRegressionGroundTruth) {
        const ranges = this.regressionService.ranges[`${this.model}:${key}`];
        const yRange = ranges.error;
        if (yRange[0] < 0 && yRange[1] > 0) {
          d3.select(scatterplot)
              .append('line')
              .attr('x1', PredictionScoreModule.plotLeftMargin)
              .attr(
                  'y1',
                  (this.plotHeight - PredictionScoreModule.plotBottomMargin) /
                      2)
              .attr('x2', this.plotWidth + PredictionScoreModule.plotLeftMargin)
              .attr(
                  'y2',
                  (this.plotHeight - PredictionScoreModule.plotBottomMargin) /
                      2)
              .style('stroke', PredictionScoreModule.zeroLineColor);
        }
      }

      // Create a 2D brush for regression keys (since both axes are meaningful)
      // and a 1D brush for multiclass preds keys.
      const newBrush =
          (isScalarKey && hasRegressionGroundTruth) ? d3.brush() : d3.brushX();
      newBrush
          .extent([
            [0, 0],
            [
              this.plotWidth - PredictionScoreModule.plotLeftMargin,
              this.plotHeight - PredictionScoreModule.plotBottomMargin
            ]
          ])
          .on('start end', () => {
            if (!d3.event.sourceEvent) {
              return;
            }
            this.selectBrushedPoints(scatterplot);
          });

      const brushGroup =
          selected.append('g')
              .attr('id', 'brushGroup')
              .attr(
                  'transform',
                  'translate(' +
                      PredictionScoreModule.plotLeftMargin.toString() + ',0)')
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
      d3.select(scatterplot).select('#primaryOverlay').raise();

      // Create scatterplot circles.
      const circles = dataPoints.selectAll('circle')
                          .data(this.preds)
                          .enter()
                          .append('circle');

      const rngSeed = 'lit';
      // tslint:disable-next-line:no-any ban-module-namespace-object-escape
      const rng = seedrandom(rngSeed);

      circles
          .attr(
              'cx',
              (d) => {
                if (isLitSubtype(spec.output[key], 'MulticlassPreds')) {
                  const predictionLabels = spec.output[key].vocab!;
                  const index = predictionLabels.indexOf(label);
                  return xScale(d[key][index]);
                }
                // Otherwise, return the regression score.
                return xScale(d[key]);
              })
          .attr(
              'cy',
              (d) => {
                if (isLitSubtype(spec.output[key], 'Scalar') &&
                    hasRegressionGroundTruth) {
                  return yScale(d[this.regressionService.getErrorKey(key)]);
                }
                // Otherwise, return a random value.
                return yScale(rng());
              })
          .style(
              'fill',
              (d) => {
                const indexedInput =
                    this.appState.getCurrentInputDataById(d['id']);
                return this.colorService.getDatapointColor(indexedInput);
              })
          .classed('point', true)
          .on('click',
              (d) => {
                this.selectionService.selectIds([d['id']]);
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

  render() {
    // clang-format off
    return html`
      <div id='container'>
        ${this.scalarKeys.map((key) => this.renderPlot(key, ''))}
        ${this.classificationKeys.map((key) => this.renderClassificationGroup(key))}
      </div>
    `;
    // clang-format on
  }

  renderClassificationGroup(key: string) {
    const spec = this.appState.getModelSpec(this.model);
    const predictionLabels = spec.output[key].vocab!;
    const margins = this.classificationService.marginSettings[this.model] || {};

    // In the binary classification case, only render one plot that
    // displays the positive class.
    const nullIdx = spec.output[key].null_idx;
    if (predictionLabels.length === 2 && nullIdx != null) {
      return html`<div>${this.renderThresholdSlider(margins, key)}</div>
          ${this.renderPlot(key, predictionLabels[1 - nullIdx])}`;
    }

    // Otherwise, return one plot per label in the multiclass case.
    // clang-format off
    return html`
        ${(predictionLabels != null && nullIdx != null) ?
          this.renderMarginSlider(margins, key) : null}
        ${predictionLabels.map((label) => this.renderPlot(key, label))}`;
    // clang-format on
  }

  renderPlot(key: string, label: string) {
    const axisTitle = label ? `${key}:${label}` : key;
    // clang-format off
    return html`
        <div class='plot-holder'>${svg`
          <svg class='scatterplot' data-key='${key}'
                                   data-label='${label}'
                                   data-axisTitle='${axisTitle}'>
          </svg>`}
        </div>
      `;
    // clang-format on
  }

  renderThresholdSlider(margins: NumericSetting, key: string) {
    // Convert between margin and classification threshold when displaying
    // margin as a threshold, as is done for binary classifiers.
    // Threshold is between 0 and 1 and represents the minimum score of the
    // positive (non-null) class before a datapoint is classified as positive.
    // A margin of 0 is the same as a threshold of .5 - meaning we take the
    // argmax class. A negative margin is a threshold below .5. Margin ranges
    // from -5 to 5, and can be converted the threshold through the equation
    // margin = ln(threshold / (1 - threshold)).
    const onChange = (e: Event) => {
      const newThresh = +(e.target as HTMLInputElement).value;
      const newMargin = newThresh !== 1 ?
          (newThresh !== 0 ? Math.log(newThresh / (1 - newThresh)) : -5) :
          5;
      margins[key] = newMargin;
    };
    const marginToVal = (margin: number) => {
      const val = getThresholdFromMargin(+margin);
      return Math.round(100 * val) / 100;
    };
    return this.renderSlider(
        margins, key, 0, 1, 0.01, onChange, marginToVal, 'threshold');
  }

  renderMarginSlider(margins: NumericSetting, key: string) {
    const onChange = (e: Event) => {
      const newMargin = (e.target as HTMLInputElement).value;
      margins[key] = +newMargin;
    };
    const marginToVal = (margin: number) => margin;
    return this.renderSlider(
        margins, key, -5, 5, 0.05, onChange, marginToVal, 'margin');
  }

  renderSlider(
      margins: NumericSetting, key: string, min: number, max: number,
      step: number, onChange: (e: Event) => void,
      marginToVal: (margin: number) => number, title: string) {
    const margin = margins[key];
    if (margin == null) {
      return;
    }
    const val = marginToVal(margins[key]);
    const isDefaultValue = margins[key] === 0;
    const reset = (e: Event) => {
      margins[key] = 0;
    };
    return html`
        <div class="slider-row">
          <div>${key} ${title}:</div>
          <input type="range" min="${min}" max="${max}" step="${step}"
                 .value="${val.toString()}" class="slider"
                 @change=${onChange}>
          <div class="slider-label">${val}</div>
          <button @click=${reset} ?disabled="${isDefaultValue}">Reset</button>
        </div>`;
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, ['Scalar', 'MulticlassPreds']);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'prediction-score-module': PredictionScoreModule;
  }
}
