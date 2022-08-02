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
import {customElement, property, state} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {LitRenderConfig, LitTabGroupConfig, RenderConfig} from '../services/modules_service';
import {ModulesService} from '../services/services';

import {app} from './app';
import {LitModule} from './lit_module';
import {styles} from './modules.css';
import {LitWidget} from './widget_group';

// Width of a minimized widget group. Set to the value of
// --lit-group-header-height when :host([minimized]) in widget_group.css.
const MINIMIZED_WIDTH_PX = 36;

// Minimum width for a widget group
const MIN_GROUP_WIDTH_PX = 100;

// Width changes below this delta aren't bubbled up, to avoid unnecssary width
// recalculations.
const MIN_GROUP_WIDTH_DELTA_PX = 10;

// The following values are derived from modules.css
const COMPONENT_AREA_HPAD = 16;   // 2x components-group-holder.padding
const EXPANDER_WIDTH = 8;         // expander-drag-target.width

// Main section height types and settings
type SectionHeightPreset = 'lower' | 'split' | 'upper';
const MAIN_SECTION_HEIGHT_MIDDLE = 45;  // % of outer-container height
const MIN_TAG_GROUP_HEIGHT = 90;  // Minimum group height in px

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

  /** Percentage of .outer-container's height given to the upper tab group. */
  @property({type: Number}) mainSectionHeight = MAIN_SECTION_HEIGHT_MIDDLE;
  @observable upperLayoutWidths: LayoutWidths = {};
  @observable lowerLayoutWidths: LayoutWidths = {};

  /**
   * A dictionary containing the percentages of .outer-container's height that
   * should be allocated to the upper tab group when the user clicks a preset
   * space allocation button in the center tab bar. These preset states are:
   *
   * * `lower`: Maximize the space allocated to the lower tab group.
   * * `split`: Approximately equal allocation to both tab groups.
   * * `upper`: Maximize the space allocated to the upper tab group.
   *
   * These values are also used to set the disabled states for the tab bar
   * position preset buttons, i.e., if this.mainSectionHeight === {value} then
   * disable the associated button.
   */
  @state() private readonly upperGroupHeightPresets = Object.seal({
    lower: 0,
    split: MAIN_SECTION_HEIGHT_MIDDLE,
    upper: 100
  });

  private readonly resizeObserver = new ResizeObserver(() => {
    const renderLayout = this.modulesService.getRenderLayout();
    this.calculateAllWidths(renderLayout);
    // Set offset for maximized modules. This module doesn't know which
    // toolbars are present, but we can just find the bounding area
    // explicitly.
    const container =
        this.shadowRoot!.querySelector<HTMLElement>('.outer-container')!;
    const {top, height} = container.getBoundingClientRect();
    container.style.setProperty('--top-toolbar-offset', `${top}px`);
    container.style.setProperty('--modules-area-height', `${height}px`);

    // Since the percentages associated with the preset states for maximizing
    // the upper and lower tab group areas depend on the height of
    // .outer-container, we need to update these values when .outer-container
    // resizes.
    Object.assign(this.upperGroupHeightPresets, {
      lower: Math.floor(MIN_TAG_GROUP_HEIGHT / height * 100),
      upper: Math.floor((height - MIN_TAG_GROUP_HEIGHT) / height * 100)
    });
  });

  static override get styles() {
    return [sharedStyles, styles];
  }

  override firstUpdated() {
    // We set up a callback in the modulesService to allow it to explicitly
    // trigger a rerender of this component when visible modules have been
    // updated by the user. Normally we'd do this in a reactive way, but we'd
    // like as fine-grain control over layout rendering as possible.
    this.modulesService.setRenderModulesCallback(() => {
      this.requestUpdate();
    });

    const container =
        this.shadowRoot!.querySelector<HTMLElement>('.outer-container')!;
    this.resizeObserver.observe(container);

    this.reactImmediately(
      () => this.modulesService.getSetting('mainHeight'),
      (mainHeight) => {
        if (mainHeight != null) {this.mainSectionHeight = Number(mainHeight);}
      });

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
  calculatePanelWidths(panelName: string, panelConfig: RenderConfig[][],
                       layoutWidths: LayoutWidths) {
    // Get the number of minimized widget groups to calculate the total width
    // available for non-minimized widgets.
    const numMinimized = panelConfig.reduce((agg, group) => {
      return agg + (this.modulesService.isModuleGroupHidden(group[0]) ? 1 : 0);
    }, 0);
    // Use the container width so this works correctly with simple/centered
    // layouts as well as full width.
    const containerWidth = this.shadowRoot!.querySelector('.outer-container')!
                               .getBoundingClientRect()
                               .width;
    const widthAvailable = containerWidth - COMPONENT_AREA_HPAD -
        MINIMIZED_WIDTH_PX * numMinimized -
        EXPANDER_WIDTH * (panelConfig.length - 1);

    // Get the total number of columns requested for the non-minimized widget
    // groups.
    const totalCols = panelConfig.reduce((agg, group) => {
      if (this.modulesService.isModuleGroupHidden(group[0])) return agg;
      const numColsList = group.map(config => config.moduleType.numCols);
      return agg + Math.max(...numColsList);
    }, 0);

    // Set the width for each widget group based on the maximum number of
    // columns its widgets have specified and the width available.
    let totalNonMinimizedWidth = 0;
    for (let i = 0; i < panelConfig.length; i++) {
      const configGroup = panelConfig[i];
      const numColsList = configGroup.map(config => config.moduleType.numCols);
      const widthPct = Math.max(...numColsList) / totalCols;
      layoutWidths[panelName][i] = Math.round(widthPct * widthAvailable);
      if (!this.modulesService.isModuleGroupHidden(configGroup[0])) {
        totalNonMinimizedWidth += layoutWidths[panelName][i];
      }
    }

    // It's possible to overflow by a few pixels due to rounding errors above.
    // overflow: hidden will prevent this from creating evil horizontal
    // scrollbars, but it's useful to adjust the widths anyway to keep
    // the right padding nicely aligned. This works for underflow too.
    // Adjust the right-most non-minimized group.
    for (let i = panelConfig.length - 1; i >= 0; i--) {
      if (!this.modulesService.isModuleGroupHidden(panelConfig[i][0])) {
        layoutWidths[panelName][i] += (widthAvailable - totalNonMinimizedWidth);
        break;
      }
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
    const upperHeight = lowerSectionVisible ? `${this.mainSectionHeight}%` :
                                              "100%";

    const styles = styleMap({
      '--upper-height': upperHeight,
      '--upper-tab-bar-visible': `${+upperTabsVisible}`,
    });

    const {lower, split, upper} = this.upperGroupHeightPresets;

    // clang-format off
    return html`
      <div id='outer-container' class=${containerClasses} style=${styles}>
        ${upperTabsVisible ? renderUpperTabBar() : null}
        <div id='upper-group-area'>
          ${this.renderComponentGroups(layout.upper, upperTabToSelect,
                                       this.upperLayoutWidths,
                                       'widget-group-upper')}
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
            <div class="preset-buttons">
              <mwc-icon class="icon-button" title="Maximize lower area"
                        ?disabled=${lower === this.mainSectionHeight}
                        @click=${() => {this.setMainSectionHeight('lower');}}>
                vertical_align_top
              </mwc-icon>
              <mwc-icon class="icon-button" title="Split screen"
                        ?disabled=${split === this.mainSectionHeight}
                        @click=${() => {this.setMainSectionHeight('split');}}>
                vertical_align_center
              </mwc-icon>
              <mwc-icon class="icon-button" title="Maximize upper area"
                        ?disabled=${upper === this.mainSectionHeight}
                        @click=${() => {this.setMainSectionHeight('upper');}}>
                vertical_align_bottom
              </mwc-icon>
            </div>
          </div>
          <div id='lower-group-area'>
            ${this.renderComponentGroups(layout.lower, lowerTabToSelect,
                                         this.lowerLayoutWidths,
                                         'widget-group-lower')}
          </div>
        ` : null}
      </div>`;
    // clang-format on
  }

  private setMainSectionHeight(setting: SectionHeightPreset) {
    this.mainSectionHeight = this.upperGroupHeightPresets[setting];
  }

  private onBarDragged(e: DragEvent) {
    const {top, height} = this.shadowRoot!.getElementById('outer-container')!
                                          .getBoundingClientRect();

    // When the user releases the cursor after a drag, browsers sometimes fire
    // a final DragEvent at position <0,0>, so we ignore it.
    if (e.clientY === 0) return;

    const maxHeight = height - MIN_TAG_GROUP_HEIGHT;
    const cursorPosition = e.clientY + 10 - top;
    const barPoisition =
        cursorPosition < MIN_TAG_GROUP_HEIGHT ? MIN_TAG_GROUP_HEIGHT :
        cursorPosition > maxHeight ? maxHeight : cursorPosition;
    this.mainSectionHeight = Math.floor(barPoisition / height * 100);
  }

  /**
   * Render the tabbed groups of components.
   * @param layout Layout to render
   * @param tabToSelect Tab to show as selected
   */
  renderComponentGroups(
      layout: LitTabGroupConfig, tabToSelect: string,
      layoutWidths: LayoutWidths, idPrefix: string) {
    return Object.keys(layout).map((tabName, i) => {
      const configs: RenderConfig[][] = layout[tabName];
      const selected = tabToSelect === tabName;
      const classes = classMap({selected, 'components-group-holder': true});
      return html`
        <div class=${classes}>
          ${
          this.renderWidgetGroups(
              configs, tabName, layoutWidths, `${idPrefix}-${i}`, selected)}
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
      configs: RenderConfig[][], section: string, layoutWidths: LayoutWidths,
      idPrefix: string, visible: boolean) {
    // Recalculate the widget group widths when isMinimized state changes.
    const onMin = () => {
      this.calculatePanelWidths(section, configs, layoutWidths);
    };

    return configs.map((configGroup, i) => {
      const width = layoutWidths[section]? layoutWidths[section][i] : 0;
      const isLastGroup = i === configs.length - 1;
      const id = `${idPrefix}-${i}`;

      let nextShownGroupIndex = -1;

      // Try to find an open widget group to the right of this one
      for (let adj = i + 1; adj < configs.length; adj++) {
        if (!this.modulesService.isModuleGroupHidden(configs[adj][0])) {
          nextShownGroupIndex = adj;
          break;
        }
      }

      const isDraggable =
          nextShownGroupIndex > i &&
          !this.modulesService.isModuleGroupHidden(configGroup[0]);

      const expanderStyles = styleMap({
        'cursor': isDraggable ? 'ew-resize' : 'default'
      });

      const dragged = (e: DragEvent) => {
        // If this is the rightmost group, or this isn't draggable, or this is
        // minimized, do nothing.
        if (isLastGroup || !isDraggable ||
            this.modulesService.isModuleGroupHidden(configGroup[0])) return;

        const widgetGroup = this.shadowRoot!.querySelector(`#${id}`);
        const left = widgetGroup!.getBoundingClientRect().left;
        const dragWidth = Math.round(e.clientX - left - EXPANDER_WIDTH);
        const dragLength = dragWidth - width;

        // Groups have a minimum width, so the user can't drag any further to
        // the left than that
        const atMinimum = dragWidth <= MIN_GROUP_WIDTH_PX;
        // We enforce a minimum drag distance before requesting an update,
        // effectively a distance-based throttle for render performance
        const isSufficient = Math.abs(dragLength) > MIN_GROUP_WIDTH_DELTA_PX;
        if (atMinimum || !isSufficient) return;

        // Balance the delta in width with the next open widget to its right, so
        // if a widget is expanded, then the next open widget to its right is
        // shrunk by the same amount and vice versa.
        const oldAdjacentWidth = layoutWidths[section][nextShownGroupIndex];
        const newWidth = Math.round(oldAdjacentWidth - dragLength);
        const newAdjacentWidth = Math.max(MIN_GROUP_WIDTH_PX, newWidth);
        const deltaFromDrag = newAdjacentWidth - newWidth;
        layoutWidths[section][nextShownGroupIndex] = newAdjacentWidth;
        layoutWidths[section][i] =
            dragWidth - (newAdjacentWidth > newWidth ? deltaFromDrag : 0);

        this.requestUpdate();
      };

      // clang-format off
      return html`
        <lit-widget-group id=${id} .configGroup=${configGroup} .width=${width}
                          @widget-group-minimized-changed=${onMin}
                          ?visible=${visible}>
        </lit-widget-group>
        ${isLastGroup ? html`` : html`
            <div class="expander" style=${expanderStyles}>
              <div class="expander-drag-target" draggable=${isDraggable}
                   @drag=${(e: DragEvent) => { dragged(e); }}>
              </div>
            </div>`}`;
      // clang-format on
    });
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-modules': LitModules;
  }
}
