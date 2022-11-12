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

import {styles} from './bar_chart.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';


/**
 * Bar chart visualization component.
 */
@customElement('bar-chart')
export class BarChart extends ReactiveElement {
  @observable @property({type: Object}) scores = new Map<string, number>();
  @property({type: Number}) margin = 30;  // Default margin size.
  @property({type: Number}) width = 0;
  @property({type: Number}) height = 0;
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
    // Make x and y scales.
    const labels = Array.from(this.scores.keys());
    return d3.scaleBand().domain(labels).range([0, this.width]).padding(0.1);
  }

  private getYScale() {
    let scale = this.yScale;
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

    // Make axes and format x label text to fit.
    // tslint:disable-next-line:no-any
    const formatXLabel = function(this: any) {
      const self = d3.select(this);
      let textLength = self.node().getComputedTextLength();
      let text = self.text();
      while (textLength > (30) && text.length > 0) {
        text = text.slice(0, -1);
        self.text(text + '...');
        textLength = self.node().getComputedTextLength();
      }
    };
    const xAxis = d3.axisBottom(x).ticks(1);
    const yAxis = d3.axisLeft(y).ticks(5);
    chart.append('g')
        .attr('transform', `translate(0, ${this.height})`)
        .attr('class', 'axis')
        .call(xAxis).selectAll("text")
        .style('text-anchor', 'end')
        .attr('transform', 'rotate(-45)')
        .each(formatXLabel);
    chart.append('g').attr('class', 'axis').call(yAxis);

    // Add bars.
    chart.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('width', x.bandwidth())
        .style('fill', 'var(--lit-cyea-400)')
        // clang-format off
        .attr('height', (d) => {
          // The y-axis displays the score.
          return this.height - y(d['score']);
        })
        .attr('x',(d) => {
          // The x-axis displays the label.
          const xVal = x(d['label']);
          return xVal == null ? 0 : xVal;
        })
        .attr('y', (d) => {
          return y(d['score']);
        })
        // clang-format on
        // Show tooltip on mouseover (displays score, y-value of the bar).
        .on('mouseover',
            (d, i, e) => {
              const el = e[i];
              const display = `${d.label}, value: ${d.score.toFixed(2)}`;
              this.displayTooltip(display, el);
            })
        .on('mouseout', (d, i, e) => {
          this.hideTooltip();
        });
  }

  private displayTooltip(tooltipStr: string, element: SVGGElement) {
    const tooltip = this.shadowRoot!.querySelector('#tooltip') as HTMLElement;
    tooltip.innerText = tooltipStr;
    tooltip.style.visibility = 'visible';
    const bcr = (element).getBoundingClientRect();
    tooltip.style.left = `${(bcr.left + bcr.right) / 2}px`;
    tooltip.style.top = `${bcr.top}px`;
  }

  private hideTooltip() {
    const tooltip = this.shadowRoot!.querySelector('#tooltip') as HTMLElement;
    tooltip.style.visibility = 'hidden';
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
    'bar-chart': BarChart;
  }
}
