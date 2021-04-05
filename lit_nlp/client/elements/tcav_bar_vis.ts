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
import '@material/mwc-icon';
import '../elements/spinner';

import * as d3 from 'd3';
import {customElement, html, property} from 'lit-element';
import {observable} from 'mobx';
import {ReactiveElement} from '../lib/elements';

import {styles} from './tcav_bar_vis.css';


/**
 * TCAV Bar chart visualization component.
 */
@customElement('tcav-bar-vis')
export class TCAVBarVis extends ReactiveElement {
  @observable @property({type: Object}) scores = new Map<string, number>();
  @property({type: Number}) margin = 30;  // Default margin size.
  @property({type: Number}) width = 0;
  @property({type: Number}) height = 0;

  static get styles() {
    return [styles];
  }

  firstUpdated() {
    this.initializeChart();

    const getScores = () => this.scores;
    this.reactImmediately(getScores, scores => {
      this.initializeChart();
    });
  }

  private getXScale() {
    // Make x and y scales.
    const labels = Array.from(this.scores.keys());
    return d3.scaleBand().domain(labels).range([0, this.width]).padding(0.1);
  }

  private getYScale() {
    return d3.scaleLinear().domain([0, 1]).range([this.height, 0]);
  }

  private initializeChart() {
    /* Sets svg dimensions, adds a group for the chart, and adds line at 0.5.*/
    const canvas = this.shadowRoot!.querySelector('#chart') as SVGGElement;
    if (canvas == null) return;
    const chartSVG = d3.select(canvas)
                         .attr('width', this.width + 2 * this.margin)
                         .attr('height', this.height + 2 * this.margin);
    chartSVG.append('line')
        .attr('x1', this.margin)
        .attr('y1', (this.height) / 2 + this.margin)
        .attr('x2', this.width + this.margin)
        .attr('y2', (this.height) / 2 + this.margin)
        .style('stroke', 'gray');
    chartSVG.append('g')
        .attr('id', 'chart-group')
        .attr('transform', `translate(${this.margin}, ${this.margin})`);

    this.renderChart();
  }

  private renderChart() {
    // Format concept scores as bar chart data.
    const data = Array.from(this.scores.entries()).map((entry, i) => {
      const key = entry[0];
      const value = entry[1];
      return {'label': key, 'score': value};
    });

    // Get x and y scales.
    const x = this.getXScale();
    const y = this.getYScale();

    // Select and clear chart svg, and set dimensions.
    const chartElement =
        this.shadowRoot!.querySelector('#chart-group') as SVGGElement;
    if (chartElement == null) return;
    const chart = d3.select(chartElement);
    chart.selectAll('*').remove();

    // Make axes.
    const xAxis = d3.axisBottom(x).ticks(1);
    const yAxis = d3.axisLeft(y).ticks(10);
    chart.append('g')
        .attr('transform', `translate(0, ${this.height})`)
        .attr('class', 'axis')
        .call(xAxis);
    chart.append('g').attr('class', 'axis').call(yAxis);

    // Add bars.
    chart.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('width', x.bandwidth())
        .style('fill', '#07a3ba')
        // clang-format off
        .attr('height', (d) => {
          // The y-axis displays the TCAV score.
          return this.height - y(d['score']);
        })
        .attr('x',(d) => {
          // The x-axis displays the concept name.
          const xVal = x(d['label']);
          return xVal == null ? 0 : xVal;
        })
        .attr('y', (d) => {
          return y(d['score']);
        })
        // clang-format on
        // Show tooltip on mouseover (displays TCAV score, y-value of the bar).
        .on('mouseover',
            (d, i, e) => {
              const el = e[i];
              this.displayTooltip(d.score.toFixed(3).toString(), el);
            })
        .on('mouseout', (d, i, e) => {
          this.hideTooltip();
        });
  }

  private displayTooltip(value: string, element: SVGGElement) {
    const tooltip = this.shadowRoot!.querySelector('#tooltip') as HTMLElement;
    tooltip.innerText = value;
    tooltip.style.visibility = 'visible';
    const bcr = (element).getBoundingClientRect();
    tooltip.style.left = `${(bcr.left + bcr.right) / 2}px`;
    tooltip.style.top = `${bcr.top}px`;
  }

  private hideTooltip() {
    const tooltip = this.shadowRoot!.querySelector('#tooltip') as HTMLElement;
    tooltip.style.visibility = 'hidden';
  }

  render() {
    // clang-format off
    return html`
      <div id="holder">
          <svg id="chart"></svg>
          <div id='tooltip'>tooltip</div>
      </div>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'tcav-bar-vis': TCAVBarVis;
  }
}
