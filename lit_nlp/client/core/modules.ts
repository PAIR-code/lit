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
import '@material/mwc-icon';

import {html} from 'lit';
import {customElement, property} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {LitRenderConfig, LitTabGroupConfig, RenderConfig} from '../services/modules_service';
import {ModulesService} from '../services/services';

import {app} from './app';
import {LitModule} from './lit_module';
import {styles} from './modules.css';
import {LitWidget, MIN_GROUP_WIDTH_PX} from './widget_group';

// Width of a minimized widget group. From widget_group.css.
const MINIMIZED_WIDTH_PX = 38 + 4; /* width + padding */

const COMPONENT_AREA_HPAD = 4; /* padding pixels */

// Contains for each section (main section, or a tab), a mapping of widget
// groups to their calculated widths.
interface LayoutWidths {
  [tabName: string]: number[];
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
  @observable upperLayoutWidths: LayoutWidths = {};
  @observable lowerLayoutWidths: LayoutWidths = {};
  private resizeObserver!: ResizeObserver;

  static override get styles() {
    return styles;
  }

  override firstUpdated() {
    // We set up a callback in the modulesService to allow it to explicitly
    // trigger a rerender of this component when visible modules have been
    // updated by the user. Normally we'd do this in a reactive way, but we'd
    // like as fine-grain control over layout rendering as possible.
    this.modulesService.setRenderModulesCallback(() => {
      this.requestUpdate();
    });

    const container: HTMLElement =
        this.shadowRoot!.querySelector('.outer-container')!;

    this.resizeObserver = new ResizeObserver(() => {
      const renderLayout = this.modulesService.getRenderLayout();
      this.calculateAllWidths(renderLayout);
      // Set offset for maximized modules. This module doesn't know which
      // toolbars are present, but we can just find the bounding area
      // explicitly.
      const bcr = container.getBoundingClientRect();
      container.style.setProperty('--top-toolbar-offset', `${bcr.top}px`);
      container.style.setProperty('--modules-area-height', `${bcr.height}px`);
    });
    this.resizeObserver.observe(container);

    this.reactImmediately(
        () => this.modulesService.getRenderLayout(), renderLayout => {
          this.calculateAllWidths(renderLayout);
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

  calculateAllWidths(renderLayout: LitRenderConfig) {
    this.calculateWidths(renderLayout.upper, this.upperLayoutWidths);
    this.calculateWidths(renderLayout.lower, this.lowerLayoutWidths);
  }

  // Calculate widths of all modules in all tabs of a group.
  calculateWidths(groupLayout: LitTabGroupConfig, layoutWidths: LayoutWidths) {
    for (const panelName of Object.keys(groupLayout)) {
      layoutWidths[panelName] = [];
      // TODO: make this function just return values?
      this.calculatePanelWidths(
          panelName, groupLayout[panelName], layoutWidths);
    }
  }

  // Calculate widths of all module groups in a single panel.
  calculatePanelWidths(
      panelName: string, panelConfig: RenderConfig[][],
      layoutWidths: LayoutWidths) {
    // Get the number of minimized widget groups to calculate the total width
    // available for non-minimized widgets.
    let numMinimized = 0;
    for (const configGroup of panelConfig) {
      if (this.modulesService.isModuleGroupHidden(configGroup[0])) {
        numMinimized +=1;
      }
    }
    const containerWidth = this.shadowRoot!.querySelector('.outer-container')!
                               .getBoundingClientRect()
                               .width;
    const widthAvailable = containerWidth - COMPONENT_AREA_HPAD -
        MINIMIZED_WIDTH_PX * numMinimized;

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

    // Set the width for each widget group based on the maximum number of
    // columns it's widgets have specified and the width available.
    for (let i = 0; i < panelConfig.length; i++) {
      const configGroup = panelConfig[i];
      const numColsList = configGroup.map(config => config.moduleType.numCols);
      const width = Math.max(...numColsList) / totalCols * widthAvailable;
      layoutWidths[panelName][i] = width;
    }
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    // We clear the callback if / when the lit-modules component is removed.
    this.modulesService.setRenderModulesCallback(() => {});
  }

  override updated() {
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

  override render() {
    const layout = this.modulesService.getRenderLayout();
    const upperGroupNames = Object.keys(layout.upper);
    const lowerGroupNames = Object.keys(layout.lower);

    const containerClasses = classMap({
      'outer-container': true,
      'outer-container-centered':
          Boolean(this.modulesService.getSetting('centerPage')),
    });

    // By default, set the selected tab to the first tab.
    if (this.modulesService.selectedTabUpper === '') {
      this.modulesService.selectedTabUpper = upperGroupNames[0];
    }
    if (this.modulesService.selectedTabLower === '') {
      this.modulesService.selectedTabLower = lowerGroupNames[0];
    }

    // If the selected tab doesn't exist, then default to the first tab.
    const indexOfUpperTab =
        upperGroupNames.indexOf(this.modulesService.selectedTabUpper);
    const upperTabToSelect = indexOfUpperTab === -1 ?
        upperGroupNames[0] :
        upperGroupNames[indexOfUpperTab];
    const indexOfLowerTab =
        lowerGroupNames.indexOf(this.modulesService.selectedTabLower);
    const lowerTabToSelect = indexOfLowerTab === -1 ?
        lowerGroupNames[0] :
        lowerGroupNames[indexOfLowerTab];

    const setUpperTab = (name: string) => {
      this.modulesService.selectedTabUpper = name;
    };

    const setLowerTab = (name: string) => {
      this.modulesService.selectedTabLower = name;
    };

    const upperTabsVisible = Object.keys(layout.upper).length > 1;
    const renderUpperTabBar = () => {
      // clang-format on
      return html`
        <div class='tab-bar'>
          <div class='tabs-container'>
            ${this.renderTabs(upperGroupNames, upperTabToSelect, setUpperTab)}
          </div>
          </div>
        </div>
      `;
      // clang-format off
    };

    const lowerSectionVisible = Object.keys(layout.lower).length > 0;
    const upperHeight = lowerSectionVisible ? `${this.mainSectionHeight}vh` : "100%";

    const styles = styleMap({
      '--upper-height': upperHeight,
      '--num-tab-bars': `${upperTabsVisible ? 2 : 1}`,
    });

    // clang-format off
    return html`
      <div id='outer-container' class=${containerClasses} style=${styles}>
        ${upperTabsVisible ? renderUpperTabBar() : null}
        <div id='upper-group-area'>
          ${this.renderComponentGroups(layout.upper, upperTabToSelect,
                                       this.upperLayoutWidths)}
        </div>
        ${lowerSectionVisible ? html`
          <div class='tab-bar' id='center-bar'>
            <div class='tabs-container'>
              ${this.renderTabs(lowerGroupNames, lowerTabToSelect, setLowerTab)}
            </div>
            <div id='drag-container'>
              <mwc-icon class="drag-icon">drag_handle</mwc-icon>
              <div id='drag-handler' draggable='true'
                  @drag=${(e: DragEvent) => {this.onBarDragged(e);}}>
              </div>
            </div>
          </div>
          <div id='lower-group-area'>
            ${this.renderComponentGroups(layout.lower, lowerTabToSelect,
                                         this.lowerLayoutWidths)}
          </div>
        ` : null}
      </div>
    `;
    // clang-format on
  }

  private onBarDragged(e: DragEvent) {
    // TODO(lit-dev): compute this relative to the container, rather than using
    // vh?
    const main = this.shadowRoot!.getElementById('upper-group-area')!;
    const mainTopPos = main.getBoundingClientRect().top;
    // Sometimes Chrome will fire bad drag events, either at (0,0)
    // or jumping around a few hundred pixels off from the drag handler.
    // Detect and ignore these so the UI doesn't get messed up.
    const handlerBCR =
        this.shadowRoot!.getElementById(
                            'drag-handler')!.getBoundingClientRect();
    const yOffset = -10;
    if (e.clientY + yOffset < mainTopPos || e.clientY <= handlerBCR.top - 30 ||
        e.clientY >= handlerBCR.bottom + 30) {
      console.log('Anomalous drag event; skipping resize', e);
      return;
    }
    this.mainSectionHeight = Math.floor(
        (e.clientY + yOffset - mainTopPos) / window.innerHeight * 100);
  }

  /**
   * Render the tabbed groups of components.
   * @param layout Layout to render
   * @param tabToSelect Tab to show as selected
   */
  renderComponentGroups(
      layout: LitTabGroupConfig, tabToSelect: string,
      layoutWidths: LayoutWidths) {
    return Object.keys(layout).map(tabName => {
      const configs: RenderConfig[][] = layout[tabName];
      const selected = tabToSelect === tabName;
      const classes = classMap({selected, 'components-group-holder': true});
      return html`
        <div class=${classes}>
          ${this.renderWidgetGroups(configs, tabName, layoutWidths)}
        </div>`;
    });
  }


  /**
   * Render the tabs of the selection groups at the bottom of the layout.
   * @param tabNames Names of the tabs to render
   * @param tabToSelect Tab to show as selected
   */
  renderTabs(
      tabNames: string[], tabToSelect: string,
      setTabFn: (name: string) => void) {
    return tabNames.map((tabName) => {
      const onclick = (e: Event) => {
        setTabFn(tabName);
        e.preventDefault();
        // Need to trigger a manual update, since this class does not
        // respond automatically to mobx observables.
        this.requestUpdate();
      };
      const selected = tabToSelect === tabName;
      const classes = classMap({selected, tab: true});
      return html`<div class=${classes} @click=${onclick}>${tabName}</div>`;
    });
  }

  renderWidgetGroups(
      configs: RenderConfig[][], section: string, layoutWidths: LayoutWidths) {
    // Calllback for widget isMinimized state changes.
    const onMin = (event: Event) => {
      // Recalculate the widget group widths in this section.
      this.calculatePanelWidths(section, configs, layoutWidths);
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
            const widthChange = dragWidth - layoutWidths[section][i];
            const oldAdjacentWidth = layoutWidths[section][i + 1];
            layoutWidths[section][i + 1] =
                Math.max(MIN_GROUP_WIDTH_PX, oldAdjacentWidth - widthChange);
          }
        }

        // Set the width of the dragged widget group.
        layoutWidths[section][i] = dragWidth;

        this.requestUpdate();
      };

      const width = layoutWidths[section] ? layoutWidths[section][i] : 0;
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
