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
import {customElement, property} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {styleMap} from 'lit/directives/style-map.js';
import {observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {LitRenderConfig, LitTabGroupConfig, RenderConfig} from '../services/modules_service';
import {AppState, ModulesService} from '../services/services';

import {app} from './app';
import {LitModule} from './lit_module';
import {styles} from './modules.css';

// Main section height types and settings
type SectionHeightPreset = 'lower' | 'split' | 'upper';

// Width of a minimized widget group. Set to the value of
// --lit-group-header-height when :host([minimized]) in widget_group.css.
const MINIMIZED_WIDTH_PX = 36;

// The following values are derived from modules.css
const APP_STATUSBAR_HEIGHT = 24;  // lit-app-statusbar.height in px
const APP_TOOLBAR_HEIGHT = 77;  // lit-app-toolbar.height in px
const CENTER_BAR_HEIGHT = 34;  // #center-bar.height in px inclusing its border
const EXPANDER_WIDTH = 8;   // expander-drag-target.width
const LOWER_RIGHT_MARGIN_TOP = 8;   // #center-bar.margin-top in px
const LOWER_RIGHT_MIN_HEIGHT = 212;   // #lower-right.min-height in px

// Minimum width for a widget group
const MIN_GROUP_WIDTH_PX = 100;

// Width changes below this delta aren't bubbled up, to avoid unnecssary width
// recalculations.
const MIN_GROUP_WIDTH_DELTA_PX = 10;

/**
 * A CSS `calc()` function to allocate equal horizontal space between the left
 * and right columns, accounting for the double-wide expander between them.
 */
const LEFT_COLUMN_DEFAULT_WIDTH = `calc(50% - ${EXPANDER_WIDTH}px)`;
/**
 * A CSS `calc()` function to allocate equal vertical space to the upper-right
 * and lower-right content `<div>`s, accounting fo the `margin-top` on the
 * lower-rigth `<div>`.
 */
const RIGHT_COLUMN_MIDDLE_HEIGHT = `calc(50% - ${LOWER_RIGHT_MARGIN_TOP/2}px)`;

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
  private readonly appState = app.getService(AppState);
  private readonly modulesService = app.getService(ModulesService);

  /**
   * CSS value string for --upper-height, will be one of: a `calc()` function,
   * a percentage, or a length in px.
   */
  @property({type: String}) upperHeight = RIGHT_COLUMN_MIDDLE_HEIGHT;
  /**
   * CSS value string for --left-column-width, will be either a `calc()`
   * function or a length in px.
   */
  @observable leftColumnWidth = LEFT_COLUMN_DEFAULT_WIDTH;
  @observable leftLayoutWidths: LayoutWidths = {};
  @observable lowerLayoutWidths: LayoutWidths = {};
  @observable upperLayoutWidths: LayoutWidths = {};

  /**
   * A dictionary of CSS values representing the height that should be allocated
   * to the upper tab group, via the --upper-height variable, when the user
   * clicks a preset space allocation button in the center tab bar. These preset
   * states are:
   *
   * * `lower`: Maximize the space allocated to the lower tab group.
   * * `split`: Approximately equal allocation to both tab groups.
   * * `upper`: Maximize the space allocated to the upper tab group.
   *
   * These values are also used to set the disabled states for the tab bar
   * position preset buttons, i.e., if this.upperHeight === {value} then
   * disable the associated button.
   */
  private readonly upperGroupHeightPresets = Object.seal({
    lower: '0',
    split: RIGHT_COLUMN_MIDDLE_HEIGHT,
    upper: '100%'
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
  });

  static override get styles() {
    return [sharedStyles, styles];
  }

  override connectedCallback() {
    super.connectedCallback();
    // We set up a callback in the modulesService to allow it to explicitly
    // trigger a rerender of this component when visible modules have been
    // updated by the user. Normally we'd do this in a reactive way, but we'd
    // like as fine-grain control over layout rendering as possible.
    this.modulesService.setRenderModulesCallback(
        () => {this.requestUpdate();});

    this.reactImmediately(
      () => this.modulesService.getSetting('mainHeight'),
      (mainHeight) => {
        if (mainHeight != null) {this.upperHeight = `${mainHeight}%`;}
      });
    this.reactImmediately(
        () => this.modulesService.getSetting('leftWidth'), (leftWidth) => {
          if (leftWidth != null) {
            this.leftColumnWidth = `${leftWidth}%`;
          }
        });

    document.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        for (const e of this.shadowRoot!.querySelectorAll(
                 'lit-widget-group[maximized]')) {
          e.removeAttribute('maximized');
        }
      }
    });
  }

  override firstUpdated() {
    // This module needs to recompute the space allocated to the various LIT
    // modules in the layouts by calling this.calculateAllWidths() under two
    // conditions:
    //
    //   1. If the selected layout changes entirely or in part (e.g., by
    //      collapsing or exapnding a module), and
    //   2. If the container resizes (e.g., due to a window resize).
    //
    // Note that this.calculateAllWidths() sets the values of several observable
    // properties, so calling this.reactImmediately() here to compute the
    // initial module space allocation is inadvisable because it will schedule
    // another Lit.dev update while completing an exisitng update lifecycle.

    // A reaction to cover the first condition is added here, after the first
    // render pass, so that it won't accidentally be called before the DOM
    // elements exist and their sizes are initialized.
    this.react(
        () => this.modulesService.getRenderLayout(),
        (renderLayout) => {this.calculateAllWidths(renderLayout);});

    const container =
        this.shadowRoot!.querySelector<HTMLElement>('#upper-right')!;
    // The second condition is handled by a ResizeObserver, which calls
    // this.calculateAllWidths() inside its callback function. This call to
    // ResizeObserver.observe() will fire that callback immediately after this
    // first Lit.dev lifecycle completes, which is functionally equivalent to
    // calling this.reactImmediately() when registering the reaction above, but
    // done in a way that avoids scheudling updates during an existing update.
    this.resizeObserver.observe(container);
  }

  calculateAllWidths(renderLayout: LitRenderConfig) {
    this.calculateWidths(renderLayout.upper, this.upperLayoutWidths);
    this.calculateWidths(renderLayout.lower, this.lowerLayoutWidths);
    this.calculateWidths(
        renderLayout.left, this.leftLayoutWidths, 'left-column');
  }

  // Calculate widths of all modules in all tabs of a group.
  calculateWidths(groupLayout: LitTabGroupConfig, layoutWidths: LayoutWidths,
                  containerId = 'right-column') {
    for (const panelName of Object.keys(groupLayout)) {
      layoutWidths[panelName] = [];
      // TODO: make this function just return values?
      this.calculatePanelWidths(
          panelName, groupLayout[panelName], layoutWidths, containerId);
    }
  }

  numColsForConfig(config: RenderConfig): number {
    const base = config.moduleType.numCols;
    // Special case: if we are replicating a module horizontally for model SxS,
    // give it proportionally more space.
    // TODO(b/283175238): implement for datapoint comparison as well?
    if (config.moduleType.duplicateAsRow &&
        config.moduleType.duplicateForModelComparison) {
      return Math.max(1, this.appState.currentModels.length) * base;
    }
    return base;
  }

  // Calculate widths of all module groups in a single panel.
  calculatePanelWidths(
      panelName: string, panelConfig: RenderConfig[][],
      layoutWidths: LayoutWidths, containerId = 'right-column') {
    const container = this.shadowRoot!.querySelector(`#${containerId}`);
    if (!container) {
      console.error(`Container with id=${containerId} not in DOM`);
      return;
    }

    // Get the number of minimized widget groups to calculate the total width
    // available for non-minimized widgets.
    const numMinimized = panelConfig.reduce((agg, group) => {
      return agg + (this.modulesService.isModuleGroupHidden(group[0]) ? 1 : 0);
    }, 0);
    // Use the container width so this works correctly with simple/centered
    // layouts as well as full width.
    const containerWidth = container.getBoundingClientRect().width;
    const widthAvailable = containerWidth - MINIMIZED_WIDTH_PX * numMinimized -
                           EXPANDER_WIDTH * (panelConfig.length - 1);

    // Get the total number of columns requested for the non-minimized widget
    // groups.
    const totalCols = panelConfig.reduce((agg, group) => {
      if (this.modulesService.isModuleGroupHidden(group[0])) return agg;
      const numColsList = group.map(config => this.numColsForConfig(config));
      return agg + Math.max(...numColsList);
    }, 0);

    // Set the width for each widget group based on the maximum number of
    // columns its widgets have specified and the width available.
    let totalNonMinimizedWidth = 0;
    for (let i = 0; i < panelConfig.length; i++) {
      const configGroup = panelConfig[i];
      const numColsList =
          configGroup.map(config => this.numColsForConfig(config));
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
          widgetElement.isLoading = isLoading;
        };
      }
    });
  }

  override render() {
    const containerClasses = classMap({
      'outer-container': true,
      'outer-container-centered':
          Boolean(this.modulesService.getSetting('centerPage')),
    });

    const {
      upper: upperSection,  // Always shown, possibly with a tab bar at the top.
      lower: lowerSection,  // If shown, it always has the draggable divider.
      left: leftSection     // If shown, may also have a tab tab bar at the top.
    } = this.modulesService.getRenderLayout();

    const upperGroupNames = Object.keys(upperSection);
    const lowerGroupNames = Object.keys(lowerSection);
    const leftGroupNames = Object.keys(leftSection);

    const topTabBarsVisible =
        leftGroupNames.length > 1 || upperGroupNames.length > 1;
    const lowerSectionVisible = lowerGroupNames.length > 0;

    // If the selected tab doesn't exist, default to the first tab in the group.
    const {
      upper: upperSelected,
      lower: lowerSelected,
      left: leftSelected
    } = this.modulesService.selectedTabs;
    const upperTab = upperGroupNames.indexOf(upperSelected) !== -1 ?
        upperSelected : upperGroupNames[0];
    const lowerTab = lowerGroupNames.indexOf(lowerSelected) !== -1 ?
        lowerSelected : lowerGroupNames[0];
    const leftTab = leftGroupNames.indexOf(leftSelected) !== -1 ?
        leftSelected : leftGroupNames[0];

    // Functions for setting the various tabs
    const setUpperTab = (name: string) => {
      this.modulesService.selectedTabs.upper = name;
    };

    const setLowerTab = (name: string) => {
      this.modulesService.selectedTabs.lower = name;
    };

    const setLeftTab = (name: string) => {
      this.modulesService.selectedTabs.left = name;
    };

    const upperHeight = lowerSectionVisible ?
        this.upperHeight : this.upperGroupHeightPresets.upper;

    const styles = styleMap({
      '--upper-height': upperHeight,
      '--upper-tab-bar-visible': `${Number(topTabBarsVisible)}`,
      '--left-tab-bar-visible': `${Number(topTabBarsVisible)}`,
    });

    const renderDraggableDivider = () => {
      const {
        lower: lowerPreset,
        split: splitPreset,
        upper: upperPreset
      } = this.upperGroupHeightPresets;
      return html`<div class='tab-bar' id='center-bar'>
        <div class='tabs-container'>
          ${this.renderTabs(lowerGroupNames, lowerTab, setLowerTab)}
        </div>
        <div id='drag-container'>
          <div id='drag-handler' draggable='true'
              @drag=${(e: DragEvent) => {this.onBarDragged(e);}}>
              <mwc-icon class="drag-icon">drag_handle</mwc-icon>
          </div>
        </div>
        <div class="preset-buttons">
          <lit-tooltip content="Maximize lower area" tooltipPosition="left">
            <mwc-icon class="icon-button" slot="tooltip-anchor"
                      ?disabled=${lowerPreset === this.upperHeight}
                      @click=${() => {this.setUpperHeight('lower');}}>
              vertical_align_top
            </mwc-icon>
          </lit-tooltip>
          <lit-tooltip content="Split screen" tooltipPosition="left">
            <mwc-icon class="icon-button" slot="tooltip-anchor"
                      ?disabled=${splitPreset === this.upperHeight}
                      @click=${() => {this.setUpperHeight('split');}}>
              vertical_align_center
            </mwc-icon>
          </lit-tooltip>
          <lit-tooltip content="Maximize upper area" tooltipPosition="left">
            <mwc-icon class="icon-button" slot="tooltip-anchor"
                      ?disabled=${upperPreset === this.upperHeight}
                      @click=${() => {this.setUpperHeight('upper');}}>
              vertical_align_bottom
            </mwc-icon>
          </lit-tooltip>
        </div>
      </div>`;
    };

    const columnSeparatorDrag = (event: DragEvent) => {
      event.stopPropagation();
      event.preventDefault();

      const {clientX} = event;
      // When the user releases the cursor after a drag, browsers sometimes fire
      // a final DragEvent at position <0,0>, so we ignore it.
      if (clientX != null && clientX > 0) {
        // The expander between the left an right columns is double width
        // compared to the expanders between module widgets; the extra .5 puts
        // the mouse in the middle of the expander once the mouse stops moving.
        this.leftColumnWidth = `${clientX - 2.5 * EXPANDER_WIDTH}px`;
      }
    };

    const columnSeparatorDoubleClick = (event: DragEvent) => {
      event.stopPropagation();
      event.preventDefault();
      const layoutDefaultLeftWidth =
          this.modulesService.getSetting('leftWidth');
      if (layoutDefaultLeftWidth != null) {
        this.leftColumnWidth = `${layoutDefaultLeftWidth}%`;
      } else {
        this.leftColumnWidth = LEFT_COLUMN_DEFAULT_WIDTH;
      }
    };

    const leftColumnStyles = styleMap({
      '--left-column-width': this.leftColumnWidth
    });

    // clang-format off
    return html`
      <div id="outer-container" class=${containerClasses} style=${styles}>
        ${leftGroupNames.length > 0 ?
            html`<div id="left-column" class="group-area"
              style=${leftColumnStyles}>
              ${topTabBarsVisible ?
                  this.renderTabBar(leftGroupNames, leftTab, setLeftTab) : null}
              ${this.renderComponentGroups(
                  leftSection, leftTab, this.leftLayoutWidths,
                  'widget-group-left', 'left-column')}
            </div>
            <div class="expander" style="cursor: ew-resize; width: 16px;">
              <div class="expander-drag-target" draggable="true"
                @drag=${columnSeparatorDrag}
                @dblclick=${columnSeparatorDoubleClick}></div>
            </div>` :
            null}
        <div id="right-column">
          <div id="upper-right" class="group-area">
            ${topTabBarsVisible ?
                this.renderTabBar(upperGroupNames, upperTab, setUpperTab) :
                null}
            ${this.renderComponentGroups(
                upperSection, upperTab, this.upperLayoutWidths,
                'widget-group-upper')}
          </div>
          ${lowerSectionVisible ?
              html`<div id="lower-right" class="group-area">
                ${renderDraggableDivider()}
                ${this.renderComponentGroups(
                    lowerSection, lowerTab, this.lowerLayoutWidths,
                    'widget-group-lower')}
              </div>` :
              null}
        </div>
      </div>`;
    // clang-format on
  }

  private setUpperHeight(setting: SectionHeightPreset) {
    // SectionHeightPreset is guaranteed to be a keyof upperGroupHeightPresets.
    // tslint:disable-next-line:no-dict-access-on-struct-type
    this.upperHeight = this.upperGroupHeightPresets[setting];
  }

  private onBarDragged(event: DragEvent) {
    event.stopPropagation();
    event.preventDefault();
    const {clientY} = event;
    // When the user releases the cursor after a drag, browsers sometimes fire
    // a final DragEvent at position <0,0>, so we ignore it.
    if (clientY === 0) return;

    const container = this.shadowRoot!.getElementById('outer-container');
    if (!container) return;

    // Compute maximum possible value of #upper-right.height as a number in px.
    const maxUpperRightHeight =
        window.innerHeight - APP_STATUSBAR_HEIGHT - APP_TOOLBAR_HEIGHT -
        LOWER_RIGHT_MIN_HEIGHT - LOWER_RIGHT_MARGIN_TOP;
    // Compute an exact value for #upper-right.height, as a number in px, based
    // on the current position of the cursor, given by event.clientY.
    const upperRightHeight =
        clientY - APP_TOOLBAR_HEIGHT - CENTER_BAR_HEIGHT/2 -
        LOWER_RIGHT_MARGIN_TOP;
    // Adjust #upper-right.height to prevent potential overflow of the parent.
    const adjustedUpperRightHeight =
        upperRightHeight > maxUpperRightHeight ? maxUpperRightHeight :
        upperRightHeight < LOWER_RIGHT_MIN_HEIGHT ? 0 : upperRightHeight;
    this.upperHeight = `${adjustedUpperRightHeight}px`;
  }

  /**
   * Render the tabbed groups of components.
   * @param layout Layout to render
   * @param tabToSelect Tab to show as selected
   */
  renderComponentGroups(
      layout: LitTabGroupConfig, tabToSelect: string,
      layoutWidths: LayoutWidths, idPrefix: string,
      containerId = 'right-column') {
    return Object.keys(layout).map((tabName, i) => {
      const selected = tabToSelect === tabName;
      const classes = classMap({
        'components-group-holder': true,
        'selected': selected,
      });
      return html`<div class=${classes}>
        ${this.renderWidgetGroups(layout[tabName], tabName, layoutWidths,
                                  `${idPrefix}-${i}`, selected, containerId)}
      </div>`;
    });
  }


  /**
   * Render the tabs of the selection groups at the bottom of the layout.
   * @param tabNames Names of the tabs to render
   * @param selectedTab Tab to show as selected
   */
  renderTabs(
    tabNames: string[], selectedTab: string, setTab: (tab: string) => void) {
    return tabNames.map((tabName) => {
      const onclick = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        setTab(tabName);
        // Need to trigger a manual update, since this class does not
        // respond automatically to MobX observables.
        this.requestUpdate();
      };
      const classes = classMap({
        'selected': selectedTab === tabName, 'tab': true
      });
      return html`<div class=${classes} @click=${onclick}>${tabName}</div>`;
    });
  }

  renderTabBar(
      tabNames: string[], selectedTab: string, setTab: (tab: string) => void) {
    return html`<div class='tab-bar'>
      <div class='tabs-container'>
        ${this.renderTabs(tabNames, selectedTab, setTab)}
      </div>
    </div>`;
  }

  renderWidgetGroups(
      configs: RenderConfig[][], section: string, layoutWidths: LayoutWidths,
      idPrefix: string, visible: boolean, containerId = 'right-column') {
    // Recalculate the widget group widths when isMinimized state changes.
    const onMin = () => {
      this.calculatePanelWidths(section, configs, layoutWidths, containerId);
    };

    const allowMinimize = configs.length > 1;

    return configs.map((configGroup, i) => {
      const width = layoutWidths[section] ? layoutWidths[section][i] : 0;
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
                          ?allowMinimize=${allowMinimize} ?visible=${visible}>
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
