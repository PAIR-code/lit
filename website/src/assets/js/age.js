(function() {
  'use strict';

  const userPolymer = window.Polymer;

  /**
   * @namespace Polymer
   * @summary Polymer is a lightweight library built on top of the web
   *   standards-based Web Components API's, and makes it easy to build your
   *   own custom HTML elements.
   * @param {!PolymerInit} info Prototype for the custom element. It must contain
   *   an `is` property to specify the element name. Other properties populate
   *   the element prototype. The `properties`, `observers`, `hostAttributes`,
   *   and `listeners` properties are processed to create element features.
   * @return {!Object} Returns a custom element class for the given provided
   *   prototype `info` object. The name of the element if given by `info.is`.
   */
  window.Polymer = function(info) {
    return window.Polymer._polymerFn(info);
  };

  // support user settings on the Polymer object
  if (userPolymer) {
    Object.assign(Polymer, userPolymer);
  }

  // To be plugged by legacy implementation if loaded
  /* eslint-disable valid-jsdoc */
  /**
   * @param {!PolymerInit} info Prototype for the custom element. It must contain
   *   an `is` property to specify the element name. Other properties populate
   *   the element prototype. The `properties`, `observers`, `hostAttributes`,
   *   and `listeners` properties are processed to create element features.
   * @return {!Object} Returns a custom element class for the given provided
   *   prototype `info` object. The name of the element if given by `info.is`.
   */
  window.Polymer._polymerFn = function(info) { // eslint-disable-line no-unused-vars
    throw new Error('Load polymer.html to use the Polymer() function.');
  };
  /* eslint-enable */

  window.Polymer.version = '2.7.0';

  /* eslint-disable no-unused-vars */
  /*
  When using Closure Compiler, JSCompiler_renameProperty(property, object) is replaced by the munged name for object[property]
  We cannot alias this function, so we have to use a small shim that has the same behavior when not compiling.
  */
  window.JSCompiler_renameProperty = function(prop, obj) {
    return prop;
  };
  /* eslint-enable */

})();

(function() {
  'use strict';

  let CSS_URL_RX = /(url\()([^)]*)(\))/g;
  let ABS_URL = /(^\/)|(^#)|(^[\w-\d]*:)/;
  let workingURL;
  let resolveDoc;
  /**
   * Resolves the given URL against the provided `baseUri'.
   * 
   * Note that this function performs no resolution for URLs that start
   * with `/` (absolute URLs) or `#` (hash identifiers).  For general purpose
   * URL resolution, use `window.URL`.
   *
   * @memberof Polymer.ResolveUrl
   * @param {string} url Input URL to resolve
   * @param {?string=} baseURI Base URI to resolve the URL against
   * @return {string} resolved URL
   */
  function resolveUrl(url, baseURI) {
    if (url && ABS_URL.test(url)) {
      return url;
    }
    // Lazy feature detection.
    if (workingURL === undefined) {
      workingURL = false;
      try {
        const u = new URL('b', 'http://a');
        u.pathname = 'c%20d';
        workingURL = (u.href === 'http://a/c%20d');
      } catch (e) {
        // silently fail
      }
    }
    if (!baseURI) {
      baseURI = document.baseURI || window.location.href;
    }
    if (workingURL) {
      return (new URL(url, baseURI)).href;
    }
    // Fallback to creating an anchor into a disconnected document.
    if (!resolveDoc) {
      resolveDoc = document.implementation.createHTMLDocument('temp');
      resolveDoc.base = resolveDoc.createElement('base');
      resolveDoc.head.appendChild(resolveDoc.base);
      resolveDoc.anchor = resolveDoc.createElement('a');
      resolveDoc.body.appendChild(resolveDoc.anchor);
    }
    resolveDoc.base.href = baseURI;
    resolveDoc.anchor.href = url;
    return resolveDoc.anchor.href || url;

  }

  /**
   * Resolves any relative URL's in the given CSS text against the provided
   * `ownerDocument`'s `baseURI`.
   *
   * @memberof Polymer.ResolveUrl
   * @param {string} cssText CSS text to process
   * @param {string} baseURI Base URI to resolve the URL against
   * @return {string} Processed CSS text with resolved URL's
   */
  function resolveCss(cssText, baseURI) {
    return cssText.replace(CSS_URL_RX, function(m, pre, url, post) {
      return pre + '\'' +
        resolveUrl(url.replace(/["']/g, ''), baseURI) +
        '\'' + post;
    });
  }

  /**
   * Returns a path from a given `url`. The path includes the trailing
   * `/` from the url.
   *
   * @memberof Polymer.ResolveUrl
   * @param {string} url Input URL to transform
   * @return {string} resolved path
   */
  function pathFromUrl(url) {
    return url.substring(0, url.lastIndexOf('/') + 1);
  }

  /**
   * Module with utilities for resolving relative URL's.
   *
   * @namespace
   * @memberof Polymer
   * @summary Module with utilities for resolving relative URL's.
   */
  Polymer.ResolveUrl = {
    resolveCss: resolveCss,
    resolveUrl: resolveUrl,
    pathFromUrl: pathFromUrl
  };

})();

/** @suppress {deprecated} */
(function() {
  'use strict';

  /**
   * Sets the global, legacy settings.
   *
   * @deprecated
   * @namespace
   * @memberof Polymer
   */
  Polymer.Settings = Polymer.Settings || {};

  Polymer.Settings.useShadow = !(window.ShadyDOM);
  Polymer.Settings.useNativeCSSProperties =
    Boolean(!window.ShadyCSS || window.ShadyCSS.nativeCss);
  Polymer.Settings.useNativeCustomElements =
    !(window.customElements.polyfillWrapFlushCallback);


  /**
   * Globally settable property that is automatically assigned to
   * `Polymer.ElementMixin` instances, useful for binding in templates to
   * make URL's relative to an application's root.  Defaults to the main
   * document URL, but can be overridden by users.  It may be useful to set
   * `Polymer.rootPath` to provide a stable application mount path when
   * using client side routing.
   *
   * @memberof Polymer
   */
  Polymer.rootPath = Polymer.rootPath ||
    Polymer.ResolveUrl.pathFromUrl(document.baseURI || window.location.href);

  /**
   * Sets the global rootPath property used by `Polymer.ElementMixin` and
   * available via `Polymer.rootPath`.
   *
   * @memberof Polymer
   * @param {string} path The new root path
   * @return {void}
   */
  Polymer.setRootPath = function(path) {
    Polymer.rootPath = path;
  };

  /**
   * A global callback used to sanitize any value before inserting it into the DOM. The callback signature is:
   *
   *     Polymer = {
   *       sanitizeDOMValue: function(value, name, type, node) { ... }
   *     }
   *
   * Where:
   *
   * `value` is the value to sanitize.
   * `name` is the name of an attribute or property (for example, href).
   * `type` indicates where the value is being inserted: one of property, attribute, or text.
   * `node` is the node where the value is being inserted.
   *
   * @type {(function(*,string,string,Node):*)|undefined}
   * @memberof Polymer
   */
  Polymer.sanitizeDOMValue = Polymer.sanitizeDOMValue || null;

  /**
   * Sets the global sanitizeDOMValue available via `Polymer.sanitizeDOMValue`.
   *
   * @memberof Polymer
   * @param {(function(*,string,string,Node):*)|undefined} newSanitizeDOMValue the global sanitizeDOMValue callback
   * @return {void}
   */
  Polymer.setSanitizeDOMValue = function(newSanitizeDOMValue) {
    Polymer.sanitizeDOMValue = newSanitizeDOMValue;
  };

  /**
   * Globally settable property to make Polymer Gestures use passive TouchEvent listeners when recognizing gestures.
   * When set to `true`, gestures made from touch will not be able to prevent scrolling, allowing for smoother
   * scrolling performance.
   * Defaults to `false` for backwards compatibility.
   *
   * @memberof Polymer
   */
  Polymer.passiveTouchGestures = Polymer.passiveTouchGestures || false;

  /**
   * Sets `passiveTouchGestures` globally for all elements using Polymer Gestures.
   *
   * @memberof Polymer
   * @param {boolean} usePassive enable or disable passive touch gestures globally
   * @return {void}
   */
  Polymer.setPassiveTouchGestures = function(usePassive) {
    Polymer.passiveTouchGestures = usePassive;
  };

  Polymer.legacyOptimizations = Polymer.legacyOptimizations ||
      window.PolymerSettings && window.PolymerSettings.legacyOptimizations || false;

  /**
   * Sets `legacyOptimizations` globally for all elements. Enables
   * optimizations when only legacy Polymer() style elements are used.
   *
   * @memberof Polymer
   * @param {boolean} useLegacyOptimizations enable or disable legacy optimizations globally.
   * @return {void}
   */
  Polymer.setLegacyOptimizations = function(useLegacyOptimizations) {
    Polymer.legacyOptimizations = useLegacyOptimizations;
  };
})();


(function() {

  'use strict';
  
  // unique global id for deduping mixins.
  let dedupeId = 0;
  
  /**
   * @constructor
   * @extends {Function}
   * @private
   */
  function MixinFunction(){}
  /** @type {(WeakMap | undefined)} */
  MixinFunction.prototype.__mixinApplications;
  /** @type {(Object | undefined)} */
  MixinFunction.prototype.__mixinSet;
  
  /* eslint-disable valid-jsdoc */
  /**
   * Wraps an ES6 class expression mixin such that the mixin is only applied
   * if it has not already been applied its base argument. Also memoizes mixin
   * applications.
   *
   * @memberof Polymer
   * @template T
   * @param {T} mixin ES6 class expression mixin to wrap
   * @return {T}
   * @suppress {invalidCasts}
   */
  Polymer.dedupingMixin = function(mixin) {
    let mixinApplications = /** @type {!MixinFunction} */(mixin).__mixinApplications;
    if (!mixinApplications) {
      mixinApplications = new WeakMap();
      /** @type {!MixinFunction} */(mixin).__mixinApplications = mixinApplications;
    }
    // maintain a unique id for each mixin
    let mixinDedupeId = dedupeId++;
    function dedupingMixin(base) {
      let baseSet = /** @type {!MixinFunction} */(base).__mixinSet;
      if (baseSet && baseSet[mixinDedupeId]) {
        return base;
      }
      let map = mixinApplications;
      let extended = map.get(base);
      if (!extended) {
        extended = /** @type {!Function} */(mixin)(base);
        map.set(base, extended);
      }
      // copy inherited mixin set from the extended class, or the base class
      // NOTE: we avoid use of Set here because some browser (IE11)
      // cannot extend a base Set via the constructor.
      let mixinSet = Object.create(/** @type {!MixinFunction} */(extended).__mixinSet || baseSet || null);
      mixinSet[mixinDedupeId] = true;
      /** @type {!MixinFunction} */(extended).__mixinSet = mixinSet;
      return extended;
    }
  
    return /** @type {T} */ (dedupingMixin);
  };
  /* eslint-enable valid-jsdoc */
  
  })();
  
  
  (function() {
    'use strict';
  
    const MODULE_STYLE_LINK_SELECTOR = 'link[rel=import][type~=css]';
    const INCLUDE_ATTR = 'include';
    const SHADY_UNSCOPED_ATTR = 'shady-unscoped';
  
    function importModule(moduleId) {
      const /** Polymer.DomModule */ PolymerDomModule = customElements.get('dom-module');
      if (!PolymerDomModule) {
        return null;
      }
      return PolymerDomModule.import(moduleId);
    }
  
    function styleForImport(importDoc) {
      // NOTE: polyfill affordance.
      // under the HTMLImports polyfill, there will be no 'body',
      // but the import pseudo-doc can be used directly.
      let container = importDoc.body ? importDoc.body : importDoc;
      const importCss = Polymer.ResolveUrl.resolveCss(container.textContent,
        importDoc.baseURI);
      const style = document.createElement('style');
      style.textContent = importCss;
      return style;
    }
  
    /** @typedef {{assetpath: string}} */
    let templateWithAssetPath; // eslint-disable-line no-unused-vars
  
    /**
     * Module with utilities for collection CSS text from `<templates>`, external
     * stylesheets, and `dom-module`s.
     *
     * @namespace
     * @memberof Polymer
     * @summary Module with utilities for collection CSS text from various sources.
     */
    const StyleGather = {
  
      /**
       * Returns a list of <style> elements in a space-separated list of `dom-module`s.
       *
       * @memberof Polymer.StyleGather
       * @param {string} moduleIds List of dom-module id's within which to
       * search for css.
       * @return {!Array<!HTMLStyleElement>} Array of contained <style> elements
       * @this {StyleGather}
       */
       stylesFromModules(moduleIds) {
        const modules = moduleIds.trim().split(/\s+/);
        const styles = [];
        for (let i=0; i < modules.length; i++) {
          styles.push(...this.stylesFromModule(modules[i]));
        }
        return styles;
      },
  
      /**
       * Returns a list of <style> elements in a given `dom-module`.
       * Styles in a `dom-module` can come either from `<style>`s within the
       * first `<template>`, or else from one or more
       * `<link rel="import" type="css">` links outside the template.
       *
       * @memberof Polymer.StyleGather
       * @param {string} moduleId dom-module id to gather styles from
       * @return {!Array<!HTMLStyleElement>} Array of contained styles.
       * @this {StyleGather}
       */
      stylesFromModule(moduleId) {
        const m = importModule(moduleId);
  
        if (!m) {
          console.warn('Could not find style data in module named', moduleId);
          return [];
        }
  
        if (m._styles === undefined) {
          const styles = [];
          // module imports: <link rel="import" type="css">
          styles.push(...this._stylesFromModuleImports(m));
          // include css from the first template in the module
          const template = m.querySelector('template');
          if (template) {
            styles.push(...this.stylesFromTemplate(template,
              /** @type {templateWithAssetPath} */(m).assetpath));
          }
  
          m._styles = styles;
        }
  
        return m._styles;
      },
  
      /**
       * Returns the `<style>` elements within a given template.
       *
       * @memberof Polymer.StyleGather
       * @param {!HTMLTemplateElement} template Template to gather styles from
       * @param {string} baseURI baseURI for style content
       * @return {!Array<!HTMLStyleElement>} Array of styles
       * @this {StyleGather}
       */
      stylesFromTemplate(template, baseURI) {
        if (!template._styles) {
          const styles = [];
          // if element is a template, get content from its .content
          const e$ = template.content.querySelectorAll('style');
          for (let i=0; i < e$.length; i++) {
            let e = e$[i];
            // support style sharing by allowing styles to "include"
            // other dom-modules that contain styling
            let include = e.getAttribute(INCLUDE_ATTR);
            if (include) {
              styles.push(...this.stylesFromModules(include).filter(function(item, index, self) {
                return self.indexOf(item) === index;
              }));
            }
            if (baseURI) {
              e.textContent = Polymer.ResolveUrl.resolveCss(e.textContent, baseURI);
            }
            styles.push(e);
          }
          template._styles = styles;
        }
        return template._styles;
      },
  
      /**
       * Returns a list of <style> elements  from stylesheets loaded via `<link rel="import" type="css">` links within the specified `dom-module`.
       *
       * @memberof Polymer.StyleGather
       * @param {string} moduleId Id of `dom-module` to gather CSS from
       * @return {!Array<!HTMLStyleElement>} Array of contained styles.
       * @this {StyleGather}
       */
       stylesFromModuleImports(moduleId) {
        let m = importModule(moduleId);
        return m ? this._stylesFromModuleImports(m) : [];
      },
  
      /**
       * @memberof Polymer.StyleGather
       * @this {StyleGather}
       * @param {!HTMLElement} module dom-module element that could contain `<link rel="import" type="css">` styles
       * @return {!Array<!HTMLStyleElement>} Array of contained styles
       */
      _stylesFromModuleImports(module) {
        const styles = [];
        const p$ = module.querySelectorAll(MODULE_STYLE_LINK_SELECTOR);
        for (let i=0; i < p$.length; i++) {
          let p = p$[i];
          if (p.import) {
            const importDoc = p.import;
            const unscoped = p.hasAttribute(SHADY_UNSCOPED_ATTR);
            if (unscoped && !importDoc._unscopedStyle) {
              const style = styleForImport(importDoc);
              style.setAttribute(SHADY_UNSCOPED_ATTR, '');
              importDoc._unscopedStyle = style;
            } else if (!importDoc._style) {
              importDoc._style = styleForImport(importDoc);
            }
            styles.push(unscoped ? importDoc._unscopedStyle : importDoc._style);
          }
        }
        return styles;
      },
  
      /**
       *
       * Returns CSS text of styles in a space-separated list of `dom-module`s.
       * Note: This method is deprecated, use `stylesFromModules` instead.
       *
       * @deprecated
       * @memberof Polymer.StyleGather
       * @param {string} moduleIds List of dom-module id's within which to
       * search for css.
       * @return {string} Concatenated CSS content from specified `dom-module`s
       * @this {StyleGather}
       */
       cssFromModules(moduleIds) {
        let modules = moduleIds.trim().split(/\s+/);
        let cssText = '';
        for (let i=0; i < modules.length; i++) {
          cssText += this.cssFromModule(modules[i]);
        }
        return cssText;
      },
  
      /**
       * Returns CSS text of styles in a given `dom-module`.  CSS in a `dom-module`
       * can come either from `<style>`s within the first `<template>`, or else
       * from one or more `<link rel="import" type="css">` links outside the
       * template.
       *
       * Any `<styles>` processed are removed from their original location.
       * Note: This method is deprecated, use `styleFromModule` instead.
       *
       * @deprecated
       * @memberof Polymer.StyleGather
       * @param {string} moduleId dom-module id to gather styles from
       * @return {string} Concatenated CSS content from specified `dom-module`
       * @this {StyleGather}
       */
      cssFromModule(moduleId) {
        let m = importModule(moduleId);
        if (m && m._cssText === undefined) {
          // module imports: <link rel="import" type="css">
          let cssText = this._cssFromModuleImports(m);
          // include css from the first template in the module
          let t = m.querySelector('template');
          if (t) {
            cssText += this.cssFromTemplate(t,
              /** @type {templateWithAssetPath} */(m).assetpath);
          }
          m._cssText = cssText || null;
        }
        if (!m) {
          console.warn('Could not find style data in module named', moduleId);
        }
        return m && m._cssText || '';
      },
  
      /**
       * Returns CSS text of `<styles>` within a given template.
       *
       * Any `<styles>` processed are removed from their original location.
       * Note: This method is deprecated, use `styleFromTemplate` instead.
       *
       * @deprecated
       * @memberof Polymer.StyleGather
       * @param {!HTMLTemplateElement} template Template to gather styles from
       * @param {string} baseURI Base URI to resolve the URL against
       * @return {string} Concatenated CSS content from specified template
       * @this {StyleGather}
       */
      cssFromTemplate(template, baseURI) {
        let cssText = '';
        const e$ = this.stylesFromTemplate(template, baseURI);
        // if element is a template, get content from its .content
        for (let i=0; i < e$.length; i++) {
          let e = e$[i];
          if (e.parentNode) {
            e.parentNode.removeChild(e);
          }
          cssText += e.textContent;
        }
        return cssText;
      },
  
      /**
       * Returns CSS text from stylesheets loaded via `<link rel="import" type="css">`
       * links within the specified `dom-module`.
       *
       * Note: This method is deprecated, use `stylesFromModuleImports` instead.
       *
       * @deprecated
       *
       * @memberof Polymer.StyleGather
       * @param {string} moduleId Id of `dom-module` to gather CSS from
       * @return {string} Concatenated CSS content from links in specified `dom-module`
       * @this {StyleGather}
       */
      cssFromModuleImports(moduleId) {
        let m = importModule(moduleId);
        return m ? this._cssFromModuleImports(m) : '';
      },
  
      /**
       * @deprecated
       * @memberof Polymer.StyleGather
       * @this {StyleGather}
       * @param {!HTMLElement} module dom-module element that could contain `<link rel="import" type="css">` styles
       * @return {string} Concatenated CSS content from links in the dom-module
       */
       _cssFromModuleImports(module) {
        let cssText = '';
        let styles = this._stylesFromModuleImports(module);
        for (let i=0; i < styles.length; i++) {
          cssText += styles[i].textContent;
        }
        return cssText;
      }
    };
  
    Polymer.StyleGather = StyleGather;
  })();

  (function() {
    'use strict';
  
    let modules = {};
    let lcModules = {};
    function setModule(id, module) {
      // store id separate from lowercased id so that
      // in all cases mixedCase id will stored distinctly
      // and lowercase version is a fallback
      modules[id] = lcModules[id.toLowerCase()] = module;
    }
    function findModule(id) {
      return modules[id] || lcModules[id.toLowerCase()];
    }
  
    function styleOutsideTemplateCheck(inst) {
      if (inst.querySelector('style')) {
        console.warn('dom-module %s has style outside template', inst.id);
      }
    }
  
    /**
     * The `dom-module` element registers the dom it contains to the name given
     * by the module's id attribute. It provides a unified database of dom
     * accessible via its static `import` API.
     *
     * A key use case of `dom-module` is for providing custom element `<template>`s
     * via HTML imports that are parsed by the native HTML parser, that can be
     * relocated during a bundling pass and still looked up by `id`.
     *
     * Example:
     *
     *     <dom-module id="foo">
     *       <img src="stuff.png">
     *     </dom-module>
     *
     * Then in code in some other location that cannot access the dom-module above
     *
     *     let img = customElements.get('dom-module').import('foo', 'img');
     *
     * @customElement
     * @extends HTMLElement
     * @memberof Polymer
     * @summary Custom element that provides a registry of relocatable DOM content
     *   by `id` that is agnostic to bundling.
     * @unrestricted
     */
    class DomModule extends HTMLElement {
  
      static get observedAttributes() { return ['id']; }
  
      /**
       * Retrieves the element specified by the css `selector` in the module
       * registered by `id`. For example, this.import('foo', 'img');
       * @param {string} id The id of the dom-module in which to search.
       * @param {string=} selector The css selector by which to find the element.
       * @return {Element} Returns the element which matches `selector` in the
       * module registered at the specified `id`.
       */
      static import(id, selector) {
        if (id) {
          let m = findModule(id);
          if (m && selector) {
            return m.querySelector(selector);
          }
          return m;
        }
        return null;
      }
  
      /* eslint-disable no-unused-vars */
      /**
       * @param {string} name Name of attribute.
       * @param {?string} old Old value of attribute.
       * @param {?string} value Current value of attribute.
       * @param {?string} namespace Attribute namespace.
       * @return {void}
       */
      attributeChangedCallback(name, old, value, namespace) {
        if (old !== value) {
          this.register();
        }
      }
      /* eslint-enable no-unused-args */
  
      /**
       * The absolute URL of the original location of this `dom-module`.
       *
       * This value will differ from this element's `ownerDocument` in the
       * following ways:
       * - Takes into account any `assetpath` attribute added during bundling
       *   to indicate the original location relative to the bundled location
       * - Uses the HTMLImports polyfill's `importForElement` API to ensure
       *   the path is relative to the import document's location since
       *   `ownerDocument` is not currently polyfilled
       */
      get assetpath() {
        // Don't override existing assetpath.
        if (!this.__assetpath) {
          // note: assetpath set via an attribute must be relative to this
          // element's location; accomodate polyfilled HTMLImports
          const owner = window.HTMLImports && HTMLImports.importForElement ?
            HTMLImports.importForElement(this) || document : this.ownerDocument;
          const url = Polymer.ResolveUrl.resolveUrl(
            this.getAttribute('assetpath') || '', owner.baseURI);
          this.__assetpath = Polymer.ResolveUrl.pathFromUrl(url);
        }
        return this.__assetpath;
      }
  
      /**
       * Registers the dom-module at a given id. This method should only be called
       * when a dom-module is imperatively created. For
       * example, `document.createElement('dom-module').register('foo')`.
       * @param {string=} id The id at which to register the dom-module.
       * @return {void}
       */
      register(id) {
        id = id || this.id;
        if (id) {
          // Under strictTemplatePolicy, reject and null out any re-registered
          // dom-module since it is ambiguous whether first-in or last-in is trusted 
          if (Polymer.strictTemplatePolicy && findModule(id) !== undefined) {
            setModule(id, null);
            throw new Error(`strictTemplatePolicy: dom-module ${id} re-registered`);
          }
          this.id = id;
          setModule(id, this);
          styleOutsideTemplateCheck(this);
        }
      }
    }
  
    DomModule.prototype['modules'] = modules;
  
    customElements.define('dom-module', DomModule);
  
    /** @const */
    Polymer.DomModule = DomModule;
  
  })();

  (function() {
    'use strict';
  
    /**
     * Module with utilities for manipulating structured data path strings.
     *
     * @namespace
     * @memberof Polymer
     * @summary Module with utilities for manipulating structured data path strings.
     */
    const Path = {
  
      /**
       * Returns true if the given string is a structured data path (has dots).
       *
       * Example:
       *
       * ```
       * Polymer.Path.isPath('foo.bar.baz') // true
       * Polymer.Path.isPath('foo')         // false
       * ```
       *
       * @memberof Polymer.Path
       * @param {string} path Path string
       * @return {boolean} True if the string contained one or more dots
       */
      isPath: function(path) {
        return path.indexOf('.') >= 0;
      },
  
      /**
       * Returns the root property name for the given path.
       *
       * Example:
       *
       * ```
       * Polymer.Path.root('foo.bar.baz') // 'foo'
       * Polymer.Path.root('foo')         // 'foo'
       * ```
       *
       * @memberof Polymer.Path
       * @param {string} path Path string
       * @return {string} Root property name
       */
      root: function(path) {
        let dotIndex = path.indexOf('.');
        if (dotIndex === -1) {
          return path;
        }
        return path.slice(0, dotIndex);
      },
  
      /**
       * Given `base` is `foo.bar`, `foo` is an ancestor, `foo.bar` is not
       * Returns true if the given path is an ancestor of the base path.
       *
       * Example:
       *
       * ```
       * Polymer.Path.isAncestor('foo.bar', 'foo')         // true
       * Polymer.Path.isAncestor('foo.bar', 'foo.bar')     // false
       * Polymer.Path.isAncestor('foo.bar', 'foo.bar.baz') // false
       * ```
       *
       * @memberof Polymer.Path
       * @param {string} base Path string to test against.
       * @param {string} path Path string to test.
       * @return {boolean} True if `path` is an ancestor of `base`.
       */
      isAncestor: function(base, path) {
        //     base.startsWith(path + '.');
        return base.indexOf(path + '.') === 0;
      },
  
      /**
       * Given `base` is `foo.bar`, `foo.bar.baz` is an descendant
       *
       * Example:
       *
       * ```
       * Polymer.Path.isDescendant('foo.bar', 'foo.bar.baz') // true
       * Polymer.Path.isDescendant('foo.bar', 'foo.bar')     // false
       * Polymer.Path.isDescendant('foo.bar', 'foo')         // false
       * ```
       *
       * @memberof Polymer.Path
       * @param {string} base Path string to test against.
       * @param {string} path Path string to test.
       * @return {boolean} True if `path` is a descendant of `base`.
       */
      isDescendant: function(base, path) {
        //     path.startsWith(base + '.');
        return path.indexOf(base + '.') === 0;
      },
  
      /**
       * Replaces a previous base path with a new base path, preserving the
       * remainder of the path.
       *
       * User must ensure `path` has a prefix of `base`.
       *
       * Example:
       *
       * ```
       * Polymer.Path.translate('foo.bar', 'zot', 'foo.bar.baz') // 'zot.baz'
       * ```
       *
       * @memberof Polymer.Path
       * @param {string} base Current base string to remove
       * @param {string} newBase New base string to replace with
       * @param {string} path Path to translate
       * @return {string} Translated string
       */
      translate: function(base, newBase, path) {
        return newBase + path.slice(base.length);
      },
  
      /**
       * @param {string} base Path string to test against
       * @param {string} path Path string to test
       * @return {boolean} True if `path` is equal to `base`
       * @this {Path}
       */
      matches: function(base, path) {
        return (base === path) ||
               this.isAncestor(base, path) ||
               this.isDescendant(base, path);
      },
  
      /**
       * Converts array-based paths to flattened path.  String-based paths
       * are returned as-is.
       *
       * Example:
       *
       * ```
       * Polymer.Path.normalize(['foo.bar', 0, 'baz'])  // 'foo.bar.0.baz'
       * Polymer.Path.normalize('foo.bar.0.baz')        // 'foo.bar.0.baz'
       * ```
       *
       * @memberof Polymer.Path
       * @param {string | !Array<string|number>} path Input path
       * @return {string} Flattened path
       */
      normalize: function(path) {
        if (Array.isArray(path)) {
          let parts = [];
          for (let i=0; i<path.length; i++) {
            let args = path[i].toString().split('.');
            for (let j=0; j<args.length; j++) {
              parts.push(args[j]);
            }
          }
          return parts.join('.');
        } else {
          return path;
        }
      },
  
      /**
       * Splits a path into an array of property names. Accepts either arrays
       * of path parts or strings.
       *
       * Example:
       *
       * ```
       * Polymer.Path.split(['foo.bar', 0, 'baz'])  // ['foo', 'bar', '0', 'baz']
       * Polymer.Path.split('foo.bar.0.baz')        // ['foo', 'bar', '0', 'baz']
       * ```
       *
       * @memberof Polymer.Path
       * @param {string | !Array<string|number>} path Input path
       * @return {!Array<string>} Array of path parts
       * @this {Path}
       * @suppress {checkTypes}
       */
      split: function(path) {
        if (Array.isArray(path)) {
          return this.normalize(path).split('.');
        }
        return path.toString().split('.');
      },
  
      /**
       * Reads a value from a path.  If any sub-property in the path is `undefined`,
       * this method returns `undefined` (will never throw.
       *
       * @memberof Polymer.Path
       * @param {Object} root Object from which to dereference path from
       * @param {string | !Array<string|number>} path Path to read
       * @param {Object=} info If an object is provided to `info`, the normalized
       *  (flattened) path will be set to `info.path`.
       * @return {*} Value at path, or `undefined` if the path could not be
       *  fully dereferenced.
       * @this {Path}
       */
      get: function(root, path, info) {
        let prop = root;
        let parts = this.split(path);
        // Loop over path parts[0..n-1] and dereference
        for (let i=0; i<parts.length; i++) {
          if (!prop) {
            return;
          }
          let part = parts[i];
          prop = prop[part];
        }
        if (info) {
          info.path = parts.join('.');
        }
        return prop;
      },
  
      /**
       * Sets a value to a path.  If any sub-property in the path is `undefined`,
       * this method will no-op.
       *
       * @memberof Polymer.Path
       * @param {Object} root Object from which to dereference path from
       * @param {string | !Array<string|number>} path Path to set
       * @param {*} value Value to set to path
       * @return {string | undefined} The normalized version of the input path
       * @this {Path}
       */
      set: function(root, path, value) {
        let prop = root;
        let parts = this.split(path);
        let last = parts[parts.length-1];
        if (parts.length > 1) {
          // Loop over path parts[0..n-2] and dereference
          for (let i=0; i<parts.length-1; i++) {
            let part = parts[i];
            prop = prop[part];
            if (!prop) {
              return;
            }
          }
          // Set value to object at end of path
          prop[last] = value;
        } else {
          // Simple property set
          prop[path] = value;
        }
        return parts.join('.');
      }
  
    };
  
    /**
     * Returns true if the given string is a structured data path (has dots).
     *
     * This function is deprecated.  Use `Polymer.Path.isPath` instead.
     *
     * Example:
     *
     * ```
     * Polymer.Path.isDeep('foo.bar.baz') // true
     * Polymer.Path.isDeep('foo')         // false
     * ```
     *
     * @deprecated
     * @memberof Polymer.Path
     * @param {string} path Path string
     * @return {boolean} True if the string contained one or more dots
     */
    Path.isDeep = Path.isPath;
  
    Polymer.Path = Path;
  
  })();

  (function() {
  'use strict';

  /**
   * Module with utilities for manipulating structured data path strings.
   *
   * @namespace
   * @memberof Polymer
   * @summary Module with utilities for manipulating structured data path strings.
   */
  const Path = {

    /**
     * Returns true if the given string is a structured data path (has dots).
     *
     * Example:
     *
     * ```
     * Polymer.Path.isPath('foo.bar.baz') // true
     * Polymer.Path.isPath('foo')         // false
     * ```
     *
     * @memberof Polymer.Path
     * @param {string} path Path string
     * @return {boolean} True if the string contained one or more dots
     */
    isPath: function(path) {
      return path.indexOf('.') >= 0;
    },

    /**
     * Returns the root property name for the given path.
     *
     * Example:
     *
     * ```
     * Polymer.Path.root('foo.bar.baz') // 'foo'
     * Polymer.Path.root('foo')         // 'foo'
     * ```
     *
     * @memberof Polymer.Path
     * @param {string} path Path string
     * @return {string} Root property name
     */
    root: function(path) {
      let dotIndex = path.indexOf('.');
      if (dotIndex === -1) {
        return path;
      }
      return path.slice(0, dotIndex);
    },

    /**
     * Given `base` is `foo.bar`, `foo` is an ancestor, `foo.bar` is not
     * Returns true if the given path is an ancestor of the base path.
     *
     * Example:
     *
     * ```
     * Polymer.Path.isAncestor('foo.bar', 'foo')         // true
     * Polymer.Path.isAncestor('foo.bar', 'foo.bar')     // false
     * Polymer.Path.isAncestor('foo.bar', 'foo.bar.baz') // false
     * ```
     *
     * @memberof Polymer.Path
     * @param {string} base Path string to test against.
     * @param {string} path Path string to test.
     * @return {boolean} True if `path` is an ancestor of `base`.
     */
    isAncestor: function(base, path) {
      //     base.startsWith(path + '.');
      return base.indexOf(path + '.') === 0;
    },

    /**
     * Given `base` is `foo.bar`, `foo.bar.baz` is an descendant
     *
     * Example:
     *
     * ```
     * Polymer.Path.isDescendant('foo.bar', 'foo.bar.baz') // true
     * Polymer.Path.isDescendant('foo.bar', 'foo.bar')     // false
     * Polymer.Path.isDescendant('foo.bar', 'foo')         // false
     * ```
     *
     * @memberof Polymer.Path
     * @param {string} base Path string to test against.
     * @param {string} path Path string to test.
     * @return {boolean} True if `path` is a descendant of `base`.
     */
    isDescendant: function(base, path) {
      //     path.startsWith(base + '.');
      return path.indexOf(base + '.') === 0;
    },

    /**
     * Replaces a previous base path with a new base path, preserving the
     * remainder of the path.
     *
     * User must ensure `path` has a prefix of `base`.
     *
     * Example:
     *
     * ```
     * Polymer.Path.translate('foo.bar', 'zot', 'foo.bar.baz') // 'zot.baz'
     * ```
     *
     * @memberof Polymer.Path
     * @param {string} base Current base string to remove
     * @param {string} newBase New base string to replace with
     * @param {string} path Path to translate
     * @return {string} Translated string
     */
    translate: function(base, newBase, path) {
      return newBase + path.slice(base.length);
    },

    /**
     * @param {string} base Path string to test against
     * @param {string} path Path string to test
     * @return {boolean} True if `path` is equal to `base`
     * @this {Path}
     */
    matches: function(base, path) {
      return (base === path) ||
             this.isAncestor(base, path) ||
             this.isDescendant(base, path);
    },

    /**
     * Converts array-based paths to flattened path.  String-based paths
     * are returned as-is.
     *
     * Example:
     *
     * ```
     * Polymer.Path.normalize(['foo.bar', 0, 'baz'])  // 'foo.bar.0.baz'
     * Polymer.Path.normalize('foo.bar.0.baz')        // 'foo.bar.0.baz'
     * ```
     *
     * @memberof Polymer.Path
     * @param {string | !Array<string|number>} path Input path
     * @return {string} Flattened path
     */
    normalize: function(path) {
      if (Array.isArray(path)) {
        let parts = [];
        for (let i=0; i<path.length; i++) {
          let args = path[i].toString().split('.');
          for (let j=0; j<args.length; j++) {
            parts.push(args[j]);
          }
        }
        return parts.join('.');
      } else {
        return path;
      }
    },

    /**
     * Splits a path into an array of property names. Accepts either arrays
     * of path parts or strings.
     *
     * Example:
     *
     * ```
     * Polymer.Path.split(['foo.bar', 0, 'baz'])  // ['foo', 'bar', '0', 'baz']
     * Polymer.Path.split('foo.bar.0.baz')        // ['foo', 'bar', '0', 'baz']
     * ```
     *
     * @memberof Polymer.Path
     * @param {string | !Array<string|number>} path Input path
     * @return {!Array<string>} Array of path parts
     * @this {Path}
     * @suppress {checkTypes}
     */
    split: function(path) {
      if (Array.isArray(path)) {
        return this.normalize(path).split('.');
      }
      return path.toString().split('.');
    },

    /**
     * Reads a value from a path.  If any sub-property in the path is `undefined`,
     * this method returns `undefined` (will never throw.
     *
     * @memberof Polymer.Path
     * @param {Object} root Object from which to dereference path from
     * @param {string | !Array<string|number>} path Path to read
     * @param {Object=} info If an object is provided to `info`, the normalized
     *  (flattened) path will be set to `info.path`.
     * @return {*} Value at path, or `undefined` if the path could not be
     *  fully dereferenced.
     * @this {Path}
     */
    get: function(root, path, info) {
      let prop = root;
      let parts = this.split(path);
      // Loop over path parts[0..n-1] and dereference
      for (let i=0; i<parts.length; i++) {
        if (!prop) {
          return;
        }
        let part = parts[i];
        prop = prop[part];
      }
      if (info) {
        info.path = parts.join('.');
      }
      return prop;
    },

    /**
     * Sets a value to a path.  If any sub-property in the path is `undefined`,
     * this method will no-op.
     *
     * @memberof Polymer.Path
     * @param {Object} root Object from which to dereference path from
     * @param {string | !Array<string|number>} path Path to set
     * @param {*} value Value to set to path
     * @return {string | undefined} The normalized version of the input path
     * @this {Path}
     */
    set: function(root, path, value) {
      let prop = root;
      let parts = this.split(path);
      let last = parts[parts.length-1];
      if (parts.length > 1) {
        // Loop over path parts[0..n-2] and dereference
        for (let i=0; i<parts.length-1; i++) {
          let part = parts[i];
          prop = prop[part];
          if (!prop) {
            return;
          }
        }
        // Set value to object at end of path
        prop[last] = value;
      } else {
        // Simple property set
        prop[path] = value;
      }
      return parts.join('.');
    }

  };

  /**
   * Returns true if the given string is a structured data path (has dots).
   *
   * This function is deprecated.  Use `Polymer.Path.isPath` instead.
   *
   * Example:
   *
   * ```
   * Polymer.Path.isDeep('foo.bar.baz') // true
   * Polymer.Path.isDeep('foo')         // false
   * ```
   *
   * @deprecated
   * @memberof Polymer.Path
   * @param {string} path Path string
   * @return {boolean} True if the string contained one or more dots
   */
  Path.isDeep = Path.isPath;

  Polymer.Path = Path;

})();


/////////// JEFF /////


(function() {
  'use strict';

  /**
   * Module with utilities for manipulating structured data path strings.
   *
   * @namespace
   * @memberof Polymer
   * @summary Module with utilities for manipulating structured data path strings.
   */
  const Path = {

    /**
     * Returns true if the given string is a structured data path (has dots).
     *
     * Example:
     *
     * ```
     * Polymer.Path.isPath('foo.bar.baz') // true
     * Polymer.Path.isPath('foo')         // false
     * ```
     *
     * @memberof Polymer.Path
     * @param {string} path Path string
     * @return {boolean} True if the string contained one or more dots
     */
    isPath: function(path) {
      return path.indexOf('.') >= 0;
    },

    /**
     * Returns the root property name for the given path.
     *
     * Example:
     *
     * ```
     * Polymer.Path.root('foo.bar.baz') // 'foo'
     * Polymer.Path.root('foo')         // 'foo'
     * ```
     *
     * @memberof Polymer.Path
     * @param {string} path Path string
     * @return {string} Root property name
     */