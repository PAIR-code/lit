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

import '../elements/checkbox';
import '../elements/spinner';
import '@material/mwc-icon-button-toggle';
import '@material/mwc-icon';

import {MobxLitElement} from '@adobe/lit-mobx';
import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import { html, LitElement} from 'lit';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';

import {SCROLL_SYNC_CSS_CLASS} from '../lib/types';
import {RenderConfig} from '../services/modules_service';
import {ModulesService} from '../services/services';

import {app} from './app';
import {LitModule} from './lit_module';
import {styles as widgetStyles} from './widget.css';
import {styles as widgetGroupStyles} from './widget_group.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

/** Minimum width for a widget group. */
export const MIN_GROUP_WIDTH_PX = 100;

// Width changes below this delta aren't bubbled up, to avoid unnecssary width
// recalculations.
const MIN_GROUP_WIDTH_DELTA_PX = 10;

/**
 * Renders a group of widgets (one per model, and one per datapoint if
 * compareDatapoints is enabled) for a single component.
 */
@customElement('lit-widget-group')
export class WidgetGroup extends LitElement {
  private readonly modulesService = app.getService(ModulesService);
  @property({ type: Array }) configGroup: RenderConfig[] = [];
  @property({ type: Boolean, reflect: true }) minimized = false;
  @property({ type: Boolean, reflect: true }) maximized = false;
  @property({ type: Boolean, reflect: true }) dragging = false;
  @property({ type: Number}) width = 0;
  private widgetScrollTop = 0;
  private widgetScrollLeft = 0;
  // Not set as @observable since re-renders were not occuring when changed.
  // Instead using requestUpdate to ensure re-renders on change.
  private syncScrolling = true;

  static override get styles() {
    return [sharedStyles, widgetGroupStyles];
  }

  override firstUpdated() {
    // Set the initial minimization from modulesService.
    this.minimized = this.initMinimized();
  }

  override render() {
    return this.renderModules(this.configGroup);
  }

  /**
   * Renders the header, including the minimize/maximize logic.
   */
  renderHeader(configGroup: RenderConfig[]) {
    const title = configGroup[0].moduleType.title;

    // Maximization.
    const onMaxClick = () => {
      this.maximized = !this.maximized;
      this.setMinimized(false);
    };

    // Minimization.
    const onMinClick = () => {
      this.setMinimized(!this.minimized);
      this.maximized = false;
    };

    // Titlebar: restore if minimized, otherwise nothing.
    const onTitleClick = () => {
      if (this.minimized) {
        this.setMinimized(false);
        this.maximized = false;
      }
    };

    const renderScrollSyncControl = () => {
      const toggleSyncScrolling = () => {
        this.syncScrolling = !this.syncScrolling;
        this.requestUpdate();
      };
      return html`
        <mwc-icon-button-toggle class="icon-button scroll-toggle"
          title="Toggle scroll sync"
          onIcon="sync_disabled" offIcon="sync"
          ?on="${!this.syncScrolling}"
          @MDCIconButtonToggle:change="${toggleSyncScrolling}"
          @icon-button-toggle-change="${toggleSyncScrolling}">
        </mwc-icon-button-toggle>`;
    };

    // clang-format off
    return html`
      <div class=header>
        <div class="title" @click=${onTitleClick}>${title}</div>
        ${this.minimized || configGroup.length < 2 ?
          null :
          renderScrollSyncControl()
        }
        <mwc-icon class="icon-button min-button" @click=${onMinClick} title="Minimize">
          ${this.minimized ? 'maximize' : 'minimize'}
        </mwc-icon>
        <mwc-icon class="icon-button" @click=${onMaxClick} title="Maximize">
          ${this.maximized ? 'fullscreen_exit' : 'fullscreen'}
        </mwc-icon>
      </div>`;
    // clang-format on
  }

  /**
   * Render all modules for a given module type (one per model or datapoint).
   */
  renderModules(configGroup: RenderConfig[]) {
    const modulesInGroup = configGroup.length > 1;
    const duplicateAsRow = configGroup[0].moduleType.duplicateAsRow;

    // Set width properties based on provided width.
    const host = this.shadowRoot!.host as HTMLElement;
    const width = `${this.width}px`;
    host.style.setProperty('--width', width);
    host.style.setProperty('--min-width', width);

    const wrapperClasses = classMap({
      'wrapper': true,
      'dragging': this.dragging,
    });

    const holderClasses = classMap({
      'component-row': modulesInGroup && duplicateAsRow,
      'component-column': modulesInGroup && !duplicateAsRow,
      'holder': true,
    });

    // Set sub-component dimensions based on the container.
    const widgetStyle = {width: '100%', height: '100%'};
    if (duplicateAsRow) {
      widgetStyle['width'] = `${100 / configGroup.length}%`;
    } else {
      widgetStyle['height'] = `${100 / configGroup.length}%`;
    }
    // For clicks on the maximized-module darkened background, undo the
    // module maximization.
    const onBackgroundClick = () => {
      this.maximized = false;
    };
    // A listener to stop clicks on a maximized module from causing the
    // background click listener from firing.
    const onWrapperClick = (e: Event) => {
      e.stopPropagation();
    };

    // Omit subtitle if there's only one in the group.
    const showSubtitle = configGroup.length > 1;

    // clang-format off
    return html`
      <div class='outside' @click=${onBackgroundClick}>
        <div class=${wrapperClasses} @click=${onWrapperClick} >
          ${this.renderHeader(configGroup)}
          <div class=${holderClasses}>
            ${configGroup.map(config => this.renderModule(config, widgetStyle, showSubtitle))}
            ${this.renderExpander()}
          </div>
        </div>
       </div>
    `;
    // clang-format on
  }


  renderModule(
      config: RenderConfig, styles: {[key: string]: string},
      showSubtitle: boolean) {
    const moduleType = config.moduleType;
    const modelName = config.modelName;
    const selectionServiceIndex = config.selectionServiceIndex;

    let subtitle = modelName ?? '';
    /**
     * If defined, modules show "Main" for 0 and "Reference for 1,
     * If undefined, modules do not show selectionService related info in their
     * titles (when compare examples mode is disabled)."
     */
    if (typeof selectionServiceIndex !== 'undefined' && moduleType.duplicateForExampleComparison) {
      subtitle = subtitle.concat(`${subtitle ? ' - ' : ''} ${
        selectionServiceIndex ? 'Reference' : 'Main'}`);
    }
    // Track scolling changes to the widget and request a rerender.
    const widgetScrollCallback = (e: CustomEvent) => {
      if (this.syncScrolling) {
        this.widgetScrollLeft = e.detail['scrollLeft'];
        this.widgetScrollTop = e.detail['scrollTop'];
        this.requestUpdate();
      }
    };
    // clang-format off
    return html`
      <lit-widget
        displayTitle=${moduleType.title}
        subtitle=${showSubtitle ? subtitle : ''}
        ?highlight=${selectionServiceIndex === 1}
        @widget-scroll="${widgetScrollCallback}"
        widgetScrollLeft=${this.widgetScrollLeft}
        widgetScrollTop=${this.widgetScrollTop}
        style=${styleMap(styles)}
      >
        ${moduleType.template(modelName, selectionServiceIndex)}
      </lit-widget>
    `;
    // clang-format on
  }

  renderExpander() {

    const dragged = (e: DragEvent) => {
      const holder = this.shadowRoot!.querySelector('.holder')!;
      const left = holder.getBoundingClientRect().left;
      const dragWidth = e.clientX - left;

      if (dragWidth > MIN_GROUP_WIDTH_PX &&
          Math.abs(dragWidth - this.width) > MIN_GROUP_WIDTH_DELTA_PX) {
        const event = new CustomEvent('widget-group-drag', {
          detail: {
            dragWidth,
          }
        });
        this.dispatchEvent(event);
      }
    };

    const dragStarted  = () => {
      this.dragging = true;
    };

    if (!this.maximized && !this.minimized) {
      return html`
        <div class="expander">
          <div class="expander-drag-target" draggable='true'
            @drag=${(e: DragEvent) => { dragged(e); }}
            @dragstart=${() => { dragStarted(); }}
            @dragend=${() => { this.dragging = false; }}>
          </div>
        </div>
      `;
    } else {
      return html``;
    }
  }

  private initMinimized() {
    const config = this.configGroup[0];
    return this.modulesService.isModuleGroupHidden(config);
  }

  private setMinimized(isMinimized: boolean) {
    const config = this.configGroup[0];
    this.modulesService.toggleHiddenModule(config, isMinimized);
    this.minimized = isMinimized;
    const event = new CustomEvent('widget-group-minimized-changed', {
      detail: {
        isMinimized
      }
    });
    this.dispatchEvent(event);
  }
}



/**
 * A wrapper for a LIT module that renders the contents in a box with
 * expand/contract capabilities.
 */
@customElement('lit-widget')
export class LitWidget extends MobxLitElement {
  @property({ type: String }) displayTitle = '';
  @property({ type: String }) subtitle = '';
  @property({ type: Boolean }) isLoading = false;
  @property({ type: Boolean }) highlight = false;
  @property({ type: Number }) widgetScrollTop = 0;
  @property({ type: Number }) widgetScrollLeft = 0;

  static override get styles() {
    return [sharedStyles, widgetStyles];
  }

  override async updated() {
    // Perform this after updateComplete so that the child module has completed
    // its updated() lifecycle method before this logic is executed.
    await this.updateComplete;
    const module = this.children[0] as LitModule;

    // If a module contains an element with the SCROLL_SYNC_CSS_CLASS, then
    // scroll it appropriately and set its onscroll listener to bubble-up scroll
    // events to the parent LitWidgetGroup.
    const scrollElems = module.shadowRoot!.querySelectorAll(
        `.${SCROLL_SYNC_CSS_CLASS}`);
    if (scrollElems.length > 0) {
      scrollElems.forEach(elem => {
        const scrollElement = elem as HTMLElement;
        scrollElement.scrollTop = this.widgetScrollTop;
        scrollElement.scrollLeft = this.widgetScrollLeft;

        // Track content scrolling and pass the scrolling information back to
        // the widget group for sync'ing between duplicated widgets.
        const scrollCallback = () => {
          const event = new CustomEvent('widget-scroll', {
            detail: {
              scrollTop: scrollElement.scrollTop,
              scrollLeft: scrollElement.scrollLeft,
            }
          });
          this.dispatchEvent(event);
        };
        module.onSyncScroll = scrollCallback;
        module.requestUpdate();
      });
    } else {
      // If a module doesn't have its own scroll element set, then scroll it
      // appropriately through the content div that contains the module.
      // After render, set the scroll values for the content based on the
      // Need to set during updated() instead of render() to avoid issues
      // with scroll events interfering with synced scroll updates.
      const content = this.shadowRoot!.querySelector('.content')!;
      if (content == null) return;
      content.scrollTop = this.widgetScrollTop;
      content.scrollLeft = this.widgetScrollLeft;
    }
  }

  override render() {
    const contentClasses = classMap({
      content: true,
      loading: this.isLoading,
    });
    const holderClasses = classMap({
      holder: true,
      highlight: this.highlight,
    });
    // Track content scrolling and pass the scrolling information back to the
    // widget group for sync'ing between duplicated widgets. This covers the
    // majority of modules that don't have their own internal scrolling
    // mechanimsm tagged with the SCROLL_SYNC_CSS_CLASS.
    const scrollCallback = () => {
      const content = this.shadowRoot!.querySelector('.content')!;
      const event = new CustomEvent('widget-scroll', {
        detail: {
          scrollTop: content.scrollTop,
          scrollLeft: content.scrollLeft,
        }
      });
      this.dispatchEvent(event);
    };

    // clang-format off
    return html`
      <div class=${holderClasses}>
        ${this.subtitle ? this.renderHeader() : ''}
        <div class="container">
          ${this.isLoading ? this.renderSpinner() : null}
          <div class=${contentClasses} @scroll=${scrollCallback}>
            <slot></slot>
          </div>
        </div>
      </div>
    `;
    // clang-format on
  }

  renderSpinner() {
    return html`
      <div class="spinner-container">
        <lit-spinner size=${36} color="var(--app-secondary-color)">
        </lit-spinner>
      </div>
    `;
  }

  renderHeader() {
    return html`
    <div class=header>
      <span class="subtitle">${this.subtitle}</span>
    </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-widget': LitWidget;
  }
}
