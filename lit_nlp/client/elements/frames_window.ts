/**
 * LIT module for displaying a variable size window of image frames.
 */



import {html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './frames_window.css';

/**
 * A LIT module to display variable size list of image frames within a window.
 */
@customElement('lit-frames-window')
export class FramesWindow extends LitElement {
  @property({type: Array}) frames: string[] = [];

  static override get styles() {
    return [
      sharedStyles,
      styles,
    ];
  }


  private renderImage(imgSrc: string) {
    return html`
     <div class="frames-window-image table-image">
       <img id='frame' src="${imgSrc}"></img>
     </div>`;
  }


  override render() {
    const framesDOM =
        this.frames.map((imageSrc: string) => this.renderImage(imageSrc));

    return html`
     <div class="frames-window">
       ${framesDOM}
     </div>`;
  }
}


declare global {
  interface HTMLElementTagNameMap {
    'lit-frames-window': FramesWindow;
  }
}
