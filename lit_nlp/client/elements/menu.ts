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
import '@material/mwc-list/mwc-list-item';

import {MobxLitElement} from '@adobe/lit-mobx';
import {Menu} from '@material/mwc-menu';
import {query} from 'lit/decorators.js';
import {property} from 'lit/decorators.js';
import {customElement} from 'lit/decorators.js';
import { html, LitElement, TemplateResult} from 'lit';
import {classMap} from 'lit/directives/class-map.js';
import {styleMap} from 'lit/directives/style-map.js';
import {computed} from 'mobx';

import {styles} from './menu.css';

type ClickCallback = () => void;

const MAX_TEXT_LENGTH = 25;

/** Holds the properties for an item in the menu. */
export interface MenuItem {
  itemText: string;        // menu item text
  onClick: ClickCallback;  // callback to use for @click property
  displayIcon: boolean;    // whether to display the icon (e.g. checkmark)
  menu: MenuItem[];        // List of submenu items
  disabled: boolean;
}

/**
 * The Menu toolbar, which contains any number of menus.
 */
@customElement('lit-menu-toolbar')
export class MenuToolbar extends MobxLitElement {
  @property({type: Object}) menuData = new Map<string, MenuItem[]>();

  static override get styles() {
    return [styles];
  }

  @computed
  get menuNames(): string[] {
    return Array.from(this.menuData.keys());
  }


  override render() {
    // clang-format off
    return html`
      <div class="toolbar" id="menu-toolbar">
        ${this.menuNames.map((name) => {
          const items = this.menuData.get(name);
          if (items == null) return;
          return html`
              <lit-menu .items=${items} .name=${name}>
              </lit-menu>
              `;
        })}
      </div>
    `;
    // clang-format on
  }
}

/**
 * Dropdown menu of options, contained in the menu toolbar.
 */
@customElement('lit-menu')
export class LitMenu extends LitElement {
  @property({type: Array}) items: MenuItem[] = [];
  @property({type: String}) name = '';

  @query('#menu') menu?: Menu;
  @query('#menu_button') button?: HTMLElement;

  private readonly menuId = 'menu';
  private readonly buttonId = 'menu_button';

  private readonly openSubmenus = new Set<string>();
  private menuOpen = false;

  static override get styles() {
    return [styles];
  }

  /**
   * Returns id string for menu list item DOM element.
   */
  private getItemId(menuId: string, index: number) {
    return `${menuId}_item_${index}`;
  }

  /**
   * Returns id string for submenu DOM element.
   */
  private getSubmenuId(itemId: string) {
    return `${itemId}_menu`;
  }

  /**
   * Returns hierarchy of menu indices to track the parent menus of a submenu.
   */
  private parseId(itemId: string) {
    return itemId.split('_').filter((i) => i !== 'menu' && i !== 'item');
  }

  /**
   * Checks if itemId is a parent menu of submenuId.
   */
  private isSubmenu(itemId: string, submenuId: string) {
    const parsedItem = this.parseId(itemId);
    const parsedSubmenu = this.parseId(submenuId);
    if (parsedSubmenu.length !== parsedItem.length + 1) return false;
    return parsedSubmenu.join('_').startsWith(parsedItem.join('_'));
  }

  private anchorSubmenus(items: MenuItem[], menuId: string) {
    items.forEach((item, index) => {
      // Anchor each submenu to its item parent.
      const itemId = this.getItemId(menuId, index);
      const submenuId = this.getSubmenuId(itemId);

      // TODO(b/204677206): The following code block triggers an additonal
      // update loop after the first render because it changes the structure and
      // observed properties of the HTML elements created by this.render(). The
      // only fix for this is to re-architect this module to build the menu
      // recursively from the start, instead of rendering and then restructuring
      // the menu elements after the fact.
      const itemElement =
          this.shadowRoot!.querySelector(`#${itemId}`) as HTMLElement;
      const menu = this.shadowRoot!.querySelector(`#${submenuId}`) as Menu;
      if (menu == null || itemElement == null) return;
      menu.anchor = itemElement;

      // If this item has an additional submenu, recurse to set its anchor.
      if (item.menu.length > 0) {
        this.anchorSubmenus(item.menu, submenuId);
      }
    });
  }

  override updated() {
    // Anchor the menu to its button.
    if (this.menu == null || this.button == null) return;
    this.menu.anchor = this.button;

    // Anchor any submenus to their respective parents.
    this.anchorSubmenus(this.items, this.menuId);
  }

  renderItem(item: MenuItem, submenuId: string, itemId: string) {

    // Display the icon if displayIcon is set to true or if this item has a
    // submenu.
    const hasSubmenu = item.menu.length > 0;
    const iconStyle = styleMap({
      'visibility': (item.displayIcon || hasSubmenu ? 'visible' : 'hidden')
    });
    const itemClass = classMap({'item': true, 'disabled': item.disabled});
    const itemTextClass =
        classMap({'item-text': true, 'text-disabled': item.disabled});

    const renderMenu = () => {
      for (const id of Array.from(this.openSubmenus.keys())) {
        const menu = this.shadowRoot!.querySelector(`#${id}`) as Menu;
        if (menu == null) return;
        if (!this.isSubmenu(id, itemId)) {
          menu.close();
          this.openSubmenus.delete(id);
        }
      }

      if (!hasSubmenu) return;
      const menu = this.shadowRoot!.querySelector(`#${submenuId}`) as Menu;
      if (menu == null) return;
      menu.show();

      this.openSubmenus.add(submenuId);
    };

    // TODO(b/184549342): Consider rewriting component without Material menu
    // due to styling issues (e.g. with setting max width with
    // text-overflow:ellipses in CSS).
    // clang-format off
    return html`
      <div class=${itemClass} graphic='icon' id=${itemId} @click=${
        hasSubmenu ? null : item.onClick}
        @mouseover=${renderMenu}>
          <mwc-icon slot='graphic' class='check' style=${iconStyle}>
            ${hasSubmenu ? 'arrow_right' : 'check'}
          </mwc-icon>
          <span class=${itemTextClass}>
            ${item.itemText.slice(0, MAX_TEXT_LENGTH) +
            ((item.itemText.length > MAX_TEXT_LENGTH) ? '...': '')}
          </span>
      </div>
    `;
    // clang-format on
  }

  renderMenu(items: MenuItem[], menuId: string, submenuIndex?: number):
      TemplateResult {
    // Returns a mwc-menu containing the menu items with the passed in id.
    // Also returns any submenus contained in one of the menu items.

    // Include top offset for submenu items.
    const menuStyle = submenuIndex == null ? styleMap({}) : styleMap({
      '--submenu-index': `${submenuIndex}`,
    });

    // Since the menu is closed immediately on click events,
    // requestAnimationFrame is called to keep track of the menu's open state;
    // if the menu button is clicked to open the menu, it can check the menuOpen
    // property and calls menu.open only if the menu began in the closed state.
    const opened = () => {
      window.requestAnimationFrame(() => {
        this.menuOpen = true;
        this.requestUpdate();
      });
    };
    const closed = () => {
      window.requestAnimationFrame(() => {
        this.menuOpen = false;
        this.requestUpdate();
      });
    };

    // Closes the menu if a menu item is clicked.
    const clicked = () => {
      const menu = this.shadowRoot!.querySelector(`#${menuId}`) as Menu;
      menu.close();
    };

    // Any submenus are rendered under the same parent as the parent menu,
    // so that the submenu is positioned correctly to the right of the parent
    // (rather than over the parent).
    // clang-format off
    return html`
      <mwc-menu class='menu' style=${menuStyle} id=${menuId}
        corner=${submenuIndex == null ? 'TOP_START': 'TOP_END'}
        @click=${clicked} @opened=${submenuIndex == null ? opened: null}
        @closed=${submenuIndex == null ? closed: null} quick>
        ${items.map((item, index) => {
          const itemId = this.getItemId(menuId, index);
          const submenuId = this.getSubmenuId(itemId);
          return this.renderItem(item, submenuId, itemId);
        })}
      </mwc-menu>
      ${items.map((item, index) => {
        const itemId = this.getItemId(menuId, index);
        const submenuId = this.getSubmenuId(itemId);
        return item.menu.length > 0 ?
            this.renderMenu(item.menu, submenuId, index) : '';
      })}
    `;
    // clang-format on
  }

  override render() {
    const toggleMenu = () => {
      if (this.menu == null) return;
      if (!this.menuOpen) {
        this.menu.show();
        this.button?.focus();
      } else {
        this.button?.blur();  // Remove focus from button if the menu is closed.
      }
    };

    const buttonClass = classMap({'button-focus': this.menuOpen});
    const iconClass =
        classMap({'icon-focus': this.menuOpen, 'dropdown-icon': true});

    // clang-format off
    return html`
        <div class='menu-container'>
          <button id=${this.buttonId} class=${buttonClass} @click=${toggleMenu}>
            ${this.name}
            <span class=${iconClass} data-icon=${
                this.menuOpen ? 'arrow_drop_up' : 'arrow_drop_down'}></span>
          </button>
          <div id='menu-holder'>
            ${this.renderMenu(this.items, this.menuId)}
          </div>
        </div>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-menu': LitMenu;
    'lit-menu-toolbar': MenuToolbar;
  }
}
