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
 * LIT App selection controls toolbar
 */

import '@material/mwc-icon';
import '@material/mwc-switch';

// tslint:disable:no-new-decorators
import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement, html} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {MenuItem} from '../elements/menu';
import {compareArrays, flatten, randInt, shortenId} from '../lib/utils';
import {AppState, ColorService, SelectionService, SliceService} from '../services/services';
import {STARRED_SLICE_NAME} from '../services/slice_service';

import {styles} from './main_toolbar.css';
import {styles as sharedStyles} from './shared_styles.css';


/**
 * The selection controls toolbar
 */
@customElement('lit-main-menu')
export class LitMainMenu extends MobxLitElement {
  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly selectionService = app.getService(SelectionService);
  private readonly sliceService = app.getService(SliceService);
  private readonly colorService = app.getService(ColorService);

  private findAllParents() {
    // This may contain duplicates, but it's fine because selectionService
    // will de-duplicate automatically.
    // Note that ancestry includes the current point.
    const allAncestors = flatten(this.selectionService.selectedIds.map(
        id => this.appState.getAncestry(id)));
    return allAncestors;
  }

  private findAllChildren() {
    const ids = new Set<string>(this.selectionService.selectedIds);
    // Check all datapoints to see if they have an ancestor in the selected
    // list. If so, add them to the list.
    for (const d of this.appState.currentInputData) {
      for (const ancestorId of this.appState.getAncestry(d.id)) {
        if (this.selectionService.isIdSelected(ancestorId)) {
          ids.add(d.id);
          continue;
        }
      }
    }
    return [...ids];
  }

  private selectRelated() {
    // selectionService will de-duplicate automatically.
    this.selectionService.selectIds(
        [...this.findAllParents(), ...this.findAllChildren()]);
  }

  // Returns items for the select menu.
  getSelectItems() {
    const selectRandomDatapoint = () => {
      const nextIndex = randInt(0, this.appState.currentInputData.length);
      const nextId = this.appState.currentInputData[nextIndex].id;
      this.selectionService.selectIds([nextId]);
    };
    const selectAllRelatedCallback = () => {
      this.selectRelated();
    };
    const selectAllParentsCallback = () => {
      this.selectionService.selectIds([...this.findAllParents()]);
    };
    const selectAllChildrenCallback = () => {
      this.selectionService.selectIds([...this.findAllChildren()]);
    };
    const selectNone = () => {
      this.selectionService.selectIds([]);
    };
    const noAction = () => {};

    const hasParents = this.findAllParents().length > 1;
    const hasChildren = this.findAllChildren().length > 1;
    const hasRelatives = hasParents || hasChildren;
    const pointSelected = this.selectionService.selectedIds.length > 0;

    return [
      {
        itemText: 'Random',
        displayIcon: false,
        menu: [],
        onClick: selectRandomDatapoint,
        disabled: false,
      },
      {
        itemText: 'All related',
        displayIcon: false,
        menu: [],
        onClick: hasRelatives ? selectAllRelatedCallback :
                                noAction,  // make this check more specific
        disabled: !hasRelatives,
      },
      {
        itemText: 'Parents',
        displayIcon: false,
        menu: [],
        onClick: hasParents ? selectAllParentsCallback : noAction,
        disabled: !hasParents,
      },
      {
        itemText: 'Children',
        displayIcon: false,
        menu: [],
        onClick: hasChildren ? selectAllChildrenCallback : noAction,
        disabled: !hasChildren,
      },
      this.getSliceItem(),
      // TODO(lit-dev): Add find siblings option.
      {
        itemText: 'Clear selection',
        displayIcon: false,
        menu: [],
        onClick: selectNone,
        disabled: !pointSelected,
      },
    ];
  }

  // Returns items for the datapoint menu.
  getColorOptionItems() {
    const colorOptions = this.colorService.colorableOptions.map((option) => {
      const isSelected =
          option.name === this.colorService.selectedColorOption.name;
      return {
        itemText: option.name,
        displayIcon: isSelected,
        menu: [],
        onClick: () => {
          this.colorService.selectedColorOption = option;
        },
        disabled: false,
      };
    });

    return colorOptions;
  }

  // Returns items for the slice menu.
  getSliceItem() {
    const sliceNames = Array.from(this.sliceService.namedSlices.keys());
    const sliceItems = sliceNames.map((name) => {
      const isSelected = this.sliceService.selectedSliceName === name;
      return {
        itemText: name,
        displayIcon: isSelected,
        menu: [],
        onClick: () => {
          this.sliceService.selectNamedSlice(name);
        },
        disabled: false
      };
    });

    return {
      itemText: 'Slices',
      displayIcon: false,
      menu: sliceItems,
      onClick: () => {},
      disabled: false,
    };
  }

  // Returns menu data for the menu toolbar.
  getMenuData() {
    const menuData = new Map<string, MenuItem[]>();
    menuData.set('Select datapoint', this.getSelectItems());
    menuData.set('Color by', this.getColorOptionItems());
    return menuData;
  }

  render() {
    return html`<lit-menu-toolbar .menuData=${
        this.getMenuData()}></lit-menu-toolbar>`;
  }
}

/**
 * The selection controls toolbar
 */
@customElement('lit-main-toolbar')
export class LitMainToolbar extends MobxLitElement {
  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly selectionService = app.getService(SelectionService);
  private readonly sliceService = app.getService(SliceService);

  @observable private displayTooltip: boolean = false;

  /**
   * ID pairs, as [child, parent], from current selection or the whole dataset.
   */
  @computed
  get selectedIdPairs() {
    const data = this.selectionService.selectedOrAllInputData;
    return data.filter(d => d.meta['parentId'])
        .map(d => [d.id, d.meta['parentId']]);
  }

  private isStarred(id: string|null): boolean {
    return (id !== null) && this.sliceService.isInSlice(STARRED_SLICE_NAME, id);
  }

  private toggleStarred() {
    const primaryId = this.selectionService.primarySelectedId;
    if (primaryId == null) return;
    if (this.isStarred(primaryId)) {
      this.sliceService.removeIdsFromSlice(STARRED_SLICE_NAME, [primaryId]);
    } else {
      this.sliceService.addIdsToSlice(STARRED_SLICE_NAME, [primaryId]);
    }
  }

  /**
   * Button to select a random example.
   */
  renderLuckyButton() {
    const selectRandom = () => {
      const nextIndex = randInt(0, this.appState.currentInputData.length);
      const nextId = this.appState.currentInputData[nextIndex].id;
      this.selectionService.selectIds([nextId]);
    };
    // clang-format off
    return html`
      <button id='select-random-button' class='hairline-button'
        @click=${selectRandom}>
        <span id='select-random-text'>Select random</span>
      </button>
    `;
    // clang-format on
  }


  /**
   * Controls to page through the dataset.
   * Assume exactly one point is selected.
   */
  renderSelectionDisplay(numSelected: number, numTotal: number) {
    const primaryId = this.selectionService.primarySelectedId;
    const selectedIndex =
        this.appState.currentInputData.findIndex(ex => ex.id === primaryId);
    const numExamples = this.appState.currentInputData.length;

    // Set the selected set to the previous (-1) or next (1) example in the
    // dataset.
    const selectOffset = (offset: number) => {
      const nextIndex = (selectedIndex + offset + numExamples) % numExamples;
      const nextId = this.appState.currentInputData[nextIndex].id;
      this.selectionService.selectIds([nextId]);
    };
    const arrowsDisabled = numSelected === 0;
    const iconClass =
        classMap({'icon-button': true, 'disabled': arrowsDisabled});
    // clang-format off
    return html`
      <mwc-icon class=${iconClass} id='ds-select-prev'
        @click=${arrowsDisabled ? null: () => {selectOffset(-1);}}>
        chevron_left
      </mwc-icon>
      <div id='number-selected'>
        <span id='num-selected-text'>${numSelected}</span> of <span>${numTotal}</span> selected
      </div>
      <mwc-icon class=${iconClass} id='ds-select-next'
        @click=${arrowsDisabled ? null: () => {selectOffset(1);}}>
        chevron_right
      </mwc-icon>
    `;
    // clang-format on
  }

  /**
   * Primary selection display, including controls
   * to page the primary selection through the selected set.
   */
  renderPrimarySelectControls() {
    const numSelected = this.selectionService.selectedIds.length;
    const primaryId = this.selectionService.primarySelectedId;
    if (primaryId == null) return;

    const displayedPrimaryId = shortenId(primaryId);
    const primaryIndex = this.appState.indicesById.get(primaryId);

    const selectedIndex =
        this.selectionService.selectedIds.findIndex(id => id === primaryId);
    // Set the primary selection to the previous (-1) or next (1) example in the
    // current selected list.
    const selectOffset = (offset: number) => {
      const nextIndex = (selectedIndex + offset + numSelected) % numSelected;
      const nextId = this.selectionService.selectedIds[nextIndex];
      this.selectionService.setPrimarySelection(nextId);
    };
    // clang-format off
    return html`
      <div id='primary-selection-status' class='selection-status-group'>
        (${numSelected > 1 ? html`
          <mwc-icon class='icon-button' id='select-prev'
            @click=${() => {selectOffset(-1);}}>
            chevron_left
          </mwc-icon>
        ` : null}
        <span id='primary-text'>
          primary:&nbsp;<span class='monospace'> ${displayedPrimaryId}</span>
          ... [<span class='monospace'>${primaryIndex}</span>]
        </span>
        ${numSelected > 1 ? html`
          <mwc-icon class='icon-button' id='select-next'
            @click=${() => {selectOffset(1);}}>
            chevron_right
          </mwc-icon>
        ` : null})
      </div>
    `;
    // clang-format on
  }

  /**
   * Pair controls. Assumes reference is showing parent, main showing child.
   */
  renderPairControls() {
    const idPairs = this.selectedIdPairs;
    if (idPairs.length === 0) return null;

    const referenceSelectionService = app.getServiceArray(SelectionService)[1];
    const maybeGetIndex = (id: string|null) =>
        id != null ? this.appState.indicesById.get(id) : null;
    // ID pairs, as [child, parent]
    const primaryIds = [
      this.selectionService.primarySelectedId,
      referenceSelectionService.primarySelectedId,
    ];
    // Index pairs, as [child, parent]
    const primaryIndices = primaryIds.map(maybeGetIndex);

    // Index into idPairs, if found.
    const selectedPairIndex = primaryIds.includes(null) ?
        -1 :
        idPairs.findIndex(
            pair => (compareArrays(pair, primaryIds as string[]) === 0));
    // True if the current selection is a counterfactual pair, with the
    // reference showing the parent.
    const isPairSelected = selectedPairIndex !== -1;
    const numPairs = idPairs.length;

    // Cycle through pairs, relative to the current selected one.
    const selectOffset = (offset: number) => {
      // If no valid pair is selected (-1), ignore and start at 0.
      const start = isPairSelected ? selectedPairIndex : 0;
      const nextPairIndex = (start + offset + numPairs) % numPairs;
      // [child, parent]
      const nextIds = idPairs[nextPairIndex];
      // Special case: if no points were selected, select all the pairs
      // so that we can continue to cycle through this set.
      if (this.selectionService.selectedIds.length === 0) {
        this.selectionService.selectIds(flatten(idPairs));
      }
      // Select child in main selection, parent in reference.
      this.selectionService.setPrimarySelection(nextIds[0]);
      referenceSelectionService.setPrimarySelection(nextIds[1]);
    };

    // Note: we're using the reference selection for the parent,
    // so show that as the first element in the pair to be consistent with the
    // rest of the UI.
    // clang-format off
    return html`
      <div id='pair-controls' class='selection-status-group'>
        ${numPairs} pairs available
        <mwc-icon class='icon-button' id='select-prev'
          @click=${() => {selectOffset(-1);}}>
          chevron_left
        </mwc-icon>
        <mwc-icon class='icon-button mdi-outlined' id='select-random'
          @click=${() => {selectOffset(randInt(1, numPairs));}}>
          casino
        </mwc-icon>
        <mwc-icon class='icon-button' id='select-next'
          @click=${() => {selectOffset(1);}}>
          chevron_right
        </mwc-icon>
        ${isPairSelected ?
        html`[${primaryIndices[1]}, ${primaryIndices[0]}] selected` : null}
      </div>
    `;
    // clang-format on
  }

  renderStarButton(numSelected: number) {
    const highlightStar =
        this.isStarred(this.selectionService.primarySelectedId);

    const disabled = numSelected === 0;
    const iconClass = classMap({'icon-button': true, 'disabled': disabled});

    const starOnClick = () => {
      this.toggleStarred();
    };
    // clang-format off
    return html`
      <mwc-icon class=${iconClass} id='star-button' @click=${starOnClick}>
        ${highlightStar ? 'star' : 'star_border'}
      </mwc-icon>`;
    // clang-format on
  }

  render() {
    const clearSelection = () => {
      this.selectionService.selectIds([]);
    };
    const numSelected = this.selectionService.selectedIds.length;
    const numTotal = this.appState.currentInputData.length;
    const primaryId = this.selectionService.primarySelectedId;

    const toggleExampleComparison = () => {
      this.appState.compareExamplesEnabled =
          !this.appState.compareExamplesEnabled;
    };
    const compareDisabled =
        !this.appState.compareExamplesEnabled && numSelected === 0;
    const compareTextClass = classMap(
        {'toggle-example-comparison': true, 'text-disabled': compareDisabled});
    const tooltipStyle =
        styleMap({visibility: this.displayTooltip ? 'visible' : 'hidden'});
    const disableClicked = () => {
      this.displayTooltip = true;
    };
    const disableMouseout = () => {
      this.displayTooltip = false;
    };
    // clang-format off
    return html`
    <div class='toolbar' id='main-toolbar'>
      <div id='left-container'>
        <lit-main-menu></lit-main-menu>
        <div class='compare-container' @click=${compareDisabled ? null:
          toggleExampleComparison}
          @mouseover=${compareDisabled ?
            disableClicked: null}
          @mouseout=${compareDisabled ? disableMouseout: null}>
          <div class=${compareTextClass}>
            Compare Datapoints
          </div>
          <mwc-switch id='compare-switch'
            ?disabled='${compareDisabled}'
            .checked=${this.appState.compareExamplesEnabled}
          >
          </mwc-switch>
        </div>
        <div class='compare-tooltip tooltip' style=${tooltipStyle}>
          <mwc-icon class='tooltip-icon'>error_outline</mwc-icon>
          <span class='tooltip-text'>
            Select a datapoint to use this feature.
          </span>
        </div>
      </div>
      <div id='right-container'>
        ${primaryId !== null ? this.renderPrimarySelectControls() :  null}
        ${this.renderStarButton(numSelected)}
        ${this.renderSelectionDisplay(numSelected, numTotal)}
        ${this.appState.compareExamplesEnabled ? this.renderPairControls() : null}
        <button id="clear-selection" class="text-button" @click=${clearSelection}
          ?disabled="${numSelected === 0}">
          Clear selection
        </button>
        ${this.renderLuckyButton()}
      </div>
    </div>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-main-toolbar': LitMainToolbar;
  }
}
