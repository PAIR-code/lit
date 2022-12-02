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
import '../elements/spinner';

import * as d3 from 'd3';
import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {observable} from 'mobx';
import {ReactiveElement} from '../lib/elements';

import {styles} from './line_chart.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

/**
 * Line chart visualization component.
 */
@customElement('line-chart')
export class LineChart extends ReactiveElement {
  @observable @property({type: Object}) scores = new Map<number, number>();
  @property({type: Number}) margin = 30;  // Default margin size.
  @property({type: Number}) width = 0;
  @property({type: Number}) height = 0;
  @property({type: Array}) xScale: number[] = [];
  @property({type: Array}) yScale: number[] = [];

  static override get styles() {
    return [sharedStyles, styles];
  }

  override firstUpdated() {
    this.initializeChart();

    const getScores = () => this.scores;
    this.reactImmediately(getScores, scores => {
      this.initializeChart();
    });
  }

  private getXScale() {
    let scale: number[] = this.xScale;
    if (scale == null || scale.length < 2) {
      const labels = Array.from(this.scores.keys());
      scale = [d3.min(labels)!, d3.max(labels)!];
    }
    return d3.scaleLinear().domain(scale).range([0, this.width]);
  }

  private getYScale() {
    let scale: number[] = this.yScale;
    if (scale == null || scale.length < 2) {
      const vals = Array.from(this.scores.values());
      scale = [d3.min(vals)!, d3.max(vals)!];
    }
    return d3.scaleLinear().domain(scale).range([this.height, 0]);
  }

  private initializeChart() {
    /* Sets svg dimensions and adds a group for the chart.*/
    const canvas = this.shadowRoot!.querySelector('#chart') as SVGGElement;
    if (canvas == null) return;
    const chartSVG = d3.select(canvas)
                         .attr('width', this.width + 2 * this.margin)
                         .attr('height', this.height + 2 * this.margin);
    chartSVG.selectAll("*").remove();
    chartSVG.append('g')
        .attr('id', 'chart-group')
        .attr('transform', `translate(${this.margin}, ${this.margin})`);

    this.renderChart();
  }

  private renderChart() {
    const data = Array.from(this.scores.entries());

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
    const xAxis = d3.axisBottom(x).ticks(5);
    const yAxis = d3.axisLeft(y).ticks(5);
    chart.append('g')
        .attr('transform', `translate(0, ${this.height})`)
        .attr('class', 'axis')
        .call(xAxis);
    chart.append('g').attr('class', 'axis').call(yAxis);

    // Add line.
    chart.append('path')
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", 'var(--lit-cyea-400)')
      .attr("stroke-width", 1.5)
      .attr("d", d3.line()
        .x((d) => x(d[0]))
        .y((d) => y(d[1])));

    const focus = chart.append("g")
        .attr("class", "focus")
        .style("display", "none");

    focus.append("circle")
        .attr("r", 4)
        .attr("fill", 'var(--lit-cyea-400)')
        .attr("stroke", 'var(--lit-cyea-400)');

    const mousemove = () => {
      const xLocation = d3.mouse(this)[0] - this.margin;
      const x0 = x.invert(xLocation);
      const bisect = d3.bisect(data.map(data => data[0]), x0);
      let d = data[bisect];
      if (bisect > 0) {
        const adjacentBisect = bisect - 1;
        if (x0 - data[adjacentBisect][0] < data[bisect][0] - x0) {
          d = data[adjacentBisect];
        }
      }
      focus.attr("transform", `translate(${x(d[0])},${y(d[1])})`);

      const tooltipStr = `${d[0].toFixed(2)}, value: ${d[1].toFixed(2)}`;
      const tooltip = this.shadowRoot!.querySelector('#tooltip') as HTMLElement;
      tooltip.innerText = tooltipStr;
      tooltip.style.visibility = 'visible';
      tooltip.style.left = `${x(d[0])}px`;
      tooltip.style.top = `${y(d[1])}px`;
    };

    chart.append("rect")
        .attr("class", "overlay")
        .attr("width", this.width)
        .attr("height", this.height)
        .attr("fill", "none")
        .attr("pointer-events", "visible")
        .on("mouseover", () => { focus.style("display", null); })
        .on("mouseout", () => {
          focus.style("display", "none");
          const tooltip = this.shadowRoot!.querySelector('#tooltip') as HTMLElement;
          tooltip.style.visibility = 'hidden';
        })
        .on("mousemove", () => { mousemove(); });
  }

  override render() {
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
    'line-chart': LineChart;
  }
}
