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
 * Client-side (UI) code for the LIT tool.
 */
// tslint:disable:no-new-decorators
import {customElement, html, property} from 'lit-element';
import {observable} from 'mobx';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import '@material/mwc-icon';

import {ReactiveElement} from '../lib/elements';
import {LitRenderConfig, RenderConfig} from '../services/modules_service';
import {ModulesService} from '../services/services';

import {app} from './lit_app';
import {LitModule} from './lit_module';
import {styles} from './modules.css';
import {LitWidget, MIN_GROUP_WIDTH_PX} from './widget_group';

// Number of columns in the full width of the layout.
const NUM_COLS = 12;

// Width of a minimized widget group. From widget_group.css.
const MINIMIZED_WIDTH_PX = 36 + 2 + 8; /* width + border + padding */

// Contains for each section (main section, or a tab), a mapping of widget
// groups to their calculated widths.
interface LayoutWidths {
  [layoutSection: string]: number[];
}

/**
 * The component responsible for rendering the selected and available lit
 * modules. This component does not extend from MobxLitElement, as we want
 * to explicitly control when it rerenders (via the setRenderModulesCallback).
 */
@customElement('lit-modules')
export class LitModules extends ReactiveElement {
  private readonly modulesService = app.getService(ModulesService);
  @property({type: Number})
  mainSectionHeight = this.modulesService.getSetting('mainHeight') || 45;
  @observable layoutWidths: LayoutWidths = {};
  private resizeObserver!: ResizeObserver;

  static get styles() {
    return styles;
  }

  firstUpdated() {
    // We set up a callback in the modulesService to allow it to explicitly
    // trigger a rerender of this component when visible modules have been
    // updated by the user. Normally we'd do this in a reactive way, but we'd
    // like as fine-grain control over layout rendering as possible.
    this.modulesService.setRenderModulesCallback(() => {
      this.requestUpdate();
    });

    this.resizeObserver = new ResizeObserver(() => {
      this.calculateWidths(this.modulesService.getRenderLayout());
    });
    this.resizeObserver.observe(this);

    this.reactImmediately(
        () => this.modulesService.getRenderLayout(), renderLayout => {
      this.calculateWidths(renderLayout);
    });

    // Escape key to exit full-screen modules.
    document.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        for (const e of this.shadowRoot!.querySelectorAll(
                 'lit-widget-group[maximized]')) {
          e.removeAttribute('maximized');
        }
      }
    });
  }

  // Calculate widths of all module groups in all panels.
  calculateWidths(renderLayout: LitRenderConfig) {
    const panelNames = Object.keys(renderLayout);
    for (const panelName of panelNames) {
      this.layoutWidths[panelName] = [];
      this.calculatePanelWidths(panelName, renderLayout[panelName]);
    }
  }

  // Calculate widths of all module groups in a single panel.
  calculatePanelWidths(panelName: string, panelConfig:  RenderConfig[][]) {
    // Get the number of minimized widget groups to calculate the total width
    // available for non-minimized widgets.
    let numMinimized = 0;
    for (const configGroup of panelConfig) {
      if (this.modulesService.isModuleGroupHidden(configGroup[0])) {
        numMinimized +=1;
      }
    }
    const widthAvailable =
        window.innerWidth - MINIMIZED_WIDTH_PX * numMinimized;

    // Get the total number of columns requested for the non-minimized widget
    // groups.
    let totalCols = 0;
    for (const configGroup of panelConfig) {
      if (this.modulesService.isModuleGroupHidden(configGroup[0])) {
        continue;
      }
      const numColsList = configGroup.map(config => config.moduleType.numCols);
      totalCols += Math.max(...numColsList);
    }
    // Ensure that when a panel requests less than the full width of columns
    // that the widget groups still use up the entire width available.
    const totalColsToUse = Math.min(totalCols, NUM_COLS);

    // Set the width for each widget group based on the maximum number of
    // columns it's widgets have specified and the width available.
    for (let i = 0; i < panelConfig.length; i++) {
      const configGroup = panelConfig[i];
      const numColsList = configGroup.map(config => config.moduleType.numCols);
      const width = Math.max(...numColsList) / totalColsToUse * widthAvailable;
      this.layoutWidths[panelName][i] = width;
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    // We clear the callback if / when the lit-modules component is removed.
    this.modulesService.setRenderModulesCallback(() => {});
  }

  updated() {
    // Since the widget parent element is responsible for displaying the load
    // status of its child litModule, and we can't provide a callback to a
    // dynamically created template string component (in
    // this.renderModuleWidget), we need to imperatively set a callback on the
    // child litModule to allow it to set the loading state of its parent
    // widget.
    const widgetElements = this.shadowRoot!.querySelectorAll('lit-widget');
    widgetElements.forEach(widgetElement => {
      // Currently, a widget can only contain one LitModule, so we just select
      //  the first child and ensure it's a LitModule.
      const litModuleElement = widgetElement.children[0];
      if (litModuleElement instanceof LitModule) {
        litModuleElement.setIsLoading = (isLoading: boolean) => {
          (widgetElement as LitWidget).isLoading = isLoading;
        };
      }
    });
  }

  render() {
    const layout = this.modulesService.getRenderLayout();
    const mainPanelConfig = layout['Main'];
    const compGroupNames = Object.keys(layout).filter(k => k !== 'Main');

    // Set the centerPage custom value from settings.
    this.style.setProperty(
      'align-self',
      this.modulesService.getSetting('centerPage') ? `center` : `unset`);

    // By default, set the selected tab to the first tab.
    if (this.modulesService.selectedTab === '') {
      this.modulesService.selectedTab = compGroupNames[0];
    }

    // If the selected tab doesn't exist, then default to the first tab.
    const indexOfTab = compGroupNames.indexOf(this.modulesService.selectedTab);
    const tabToSelect = indexOfTab === -1 ? compGroupNames[0] :
        compGroupNames[indexOfTab];

    const styles = styleMap({'height': `${this.mainSectionHeight}vh`});

    // clang-format off
    return html`
      <div id='main-panel' style=${styles}>
        ${this.renderWidgetGroups(mainPanelConfig, 'Main')}
      </div>
      <div id='center-bar'>
        <div id='tabs'>
          ${this.renderTabs(compGroupNames, tabToSelect)}
        </div>
        <div id='drag-container'>
          <mwc-icon class="drag-icon">drag_handle</mwc-icon>
          <div id='drag-handler' draggable='true'
              @drag=${(e: DragEvent) => {this.onBarDragged(e);}}>
          </div>
        </div>
      </div>
      <div id='component-groups'>
        ${this.renderComponentGroups(layout, compGroupNames, tabToSelect)}
      </div>
    `;
    // clang-format on
  }

  private onBarDragged(e: DragEvent) {
    const main = this.shadowRoot!.getElementById('main-panel')!;
    const mainTopPos = main.getBoundingClientRect().top;
    this.mainSectionHeight =
        Math.floor((e.clientY - mainTopPos - 10) / window.innerHeight * 100);
  }

  /**
   * Render the tabbed groups of components.
   * @param layout Layout to render
   * @param compGroupNames Names of the components to render
   * @param tabToSelect Tab to show as selected
   */
  renderComponentGroups(layout: LitRenderConfig, compGroupNames: string[],
                        tabToSelect: string) {
    return compGroupNames.map((compGroupName) => {
      const configs = layout[compGroupName];
      const selected = tabToSelect === compGroupName;
      const classes = classMap({selected, 'components-group-holder': true});
      return html`
        <div class=${classes}>
          ${this.renderWidgetGroups(configs, compGroupName)}
        </div>`;
    });
  }


  /**
   * Render the tabs of the selection groups at the bottom of the layout.
   * @param compGroupNames Names of the tabs to render
   * @param tabToSelect Tab to show as selected
   */
  renderTabs(compGroupNames: string[], tabToSelect: string) {
    return compGroupNames.map((compGroupName) => {
      const name = compGroupName;
      const onclick = (e: Event) => {
        this.modulesService.selectedTab = name;
        e.preventDefault();
        // Need to trigger a manual update, since this class does not
        // respond automatically to mobx observables.
        this.requestUpdate();
      };
      const selected = tabToSelect === compGroupName;
      const classes = classMap({selected, tab: true});
      return html`<div class=${classes} @click=${onclick}>${
          compGroupName}</div>`;
    });
  }

  renderWidgetGroups(configs: RenderConfig[][], section: string) {
    // Calllback for widget isMinimized state changes.
    const onMin = (event: Event) => {
      // Recalculate the widget group widths in this section.
      this.calculatePanelWidths(section, configs);
    };

    return configs.map((configGroup, i) => {

      // Callback from widget width drag events.
      const onDrag = (event: Event) => {
        // tslint:disable-next-line:no-any
        const dragWidth =  (event as any).detail.dragWidth;

        // If the dragged group isn't the right-most group, then balance the
        // delta in width with the widget directly to it's left (so if a widget
        // is expanded, then its adjacent widget is shrunk by the same amount).
        if (i < configs.length - 1) {
          const adjacentConfig = configs[i + 1];
          if (!this.modulesService.isModuleGroupHidden(adjacentConfig[0])) {
            const widthChange = dragWidth -
                this.layoutWidths[section][i];
            const oldAdjacentWidth =
                this.layoutWidths[section][i + 1];
            this.layoutWidths[section][i + 1] =
                Math.max(MIN_GROUP_WIDTH_PX, oldAdjacentWidth - widthChange);
          }
        }

        // Set the width of the dragged widget group.
        this.layoutWidths[section][i] = dragWidth;

        this.requestUpdate();
      };

      const width = this.layoutWidths[section] ?
          this.layoutWidths[section][i] : 0;
      return html`<lit-widget-group .configGroup=${configGroup}
          @widget-group-minimized-changed=${onMin} @widget-group-drag=${onDrag}
          .width=${width}></lit-widget-group>`;
    });
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-modules': LitModules;
  }
}
