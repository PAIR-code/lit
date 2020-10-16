# Frontend Client



<!--* freshness: { owner: 'lit-dev' reviewed: '2020-08-04' } *-->

This document aims to describe the current LIT frontend system, including
conventions, best practices, and gotchas.



## High Level Overview

LIT is powered by two central pieces of tech -
[lit-element](https://lit-element.polymer-project.org/) for components and HTML
rendering, and [mobx](https://mobx.js.org/README.html) for observable-oriented
state management.

Lit-element is a simple, web-component based library for building small,
self-contained pieces of web functionality. It uses a template-string based
output to declaratively render small, isolated pieces of UI.

Mobx is a tool centered around observable data, and it makes managing state
simple and scalable.

We highly recommend reading the docs for both projects - they both have fairly
simple APIs and are easy to digest in comparison to some heavier-weight toolkits
like Angular.

## LIT Application Architecture

The LIT client frontend is roughly divided into three conceptual groups -
**Modules** (which render visualizations), **Services** (which manage data), and
the **App** itself (which coordinates initialization of services and determines
which modules to render).

### Bootstrapping

The LIT app bootstrapping takes place in two steps: First, the served
[`index.html`](../lit_nlp/client/static/index.html)
page contains a single web component for the
[`<lit-app>`](../lit_nlp/client/app/app.ts). This
component is responsible for the overall layout of the app, including the
toolbar, footer, and the
[`<lit-modules>`](../lit_nlp/client/core/modules.ts)
component. The `<lit-modules>` component is responsible for actually laying out
and rendering the various `LitModule` components, a process about which we'll go
into greater detail later.

The JS bundle entry point is
[`main.ts`](../lit_nlp/client/default/main.ts), which
first imports the loaded, the `<lit-app>` web component is declared, and
attaches itself to the DOM, waiting for the app to be initialized.

The second step is kicking off app initialization. The
[`LitApp`](../lit_nlp/client/core/lit_app.ts)
singleton class is provided with a layout declaring which `LitModule` components
to use, then builds the app services and kicks off app initialization and
loading data.

### Layout

A layout is defined by a structure of `LitModule` classes, and includes a set of
main components that are always visible, (designated in the object by the "main"
key) and a set of tabs that each contain a group other components.

A simplified version for a classifier model might look like:

```typescript
const layout: LitComponentLayout = {
  components : {
    'Main': [DataTableModule, DatapointEditorModule],

    'Classifiers': [
      ConfusionMatrixModule,
    ],
    'Counterfactuals': [GeneratorModule],
    'Predictions': [
      ScalarModule,
      ClassificationModule,
    ],
    'Explanations': [
      ClassificationModule,
      SalienceMapModule,
      AttentionModule,
    ]
  }
};
```

The full layouts are defined in
[layout.ts](../lit_nlp/client/default/layout.ts). To
use a specific layout for a given LIT instance, pass the key (e.g., "simple" or
"mlm") in as a server flag when initializing LIT(`--layout=<layout>`). The
layout can be set on-the-fly a URL param (the url param overrides the server
flag).

The actual layout of components in
[`<lit-modules>`](../lit_nlp/client/core/modules.ts)
can be different than the simple declared layout, since the visibility of
modules depends on a number of factors, including the user-chosen visibility and
whether or not specific modules show multiple copies per selected model. This
actual layout is computed in
[`modules_service`](../lit_nlp/client/services/modules_service.ts).

### Initialization

Finally, the LIT App initializes by building the various service classes and
starting the initial load of data from the server. This process consists of:

1.  Parsing the URL query params to get the url configuration
1.  Fetching the app metadata, which includes what models/datasets are available
    to use.
1.  Determining which models/datasets to load and then loding them.

## LIT Modules

The
[`LitModule`](../lit_nlp/client/core/lit_module.ts)
is the base class from which all module components derive. It provides a number
of convenience methods for handling common update / data loading patterns. Each
LIT Module also requires a few static methods by convention, responsible for
specifying Module display and behavior. These helpers and conventions are
outlined below:

```typescript
/**
 * A dummy module that responds to changes in selected data by making a request
 * to an API service to get the pig latin translation.
 */
@customElement('demo-module')                                                   // (0)
export class DemoTextModule extends LitModule {
  static title = 'Demo Module';                                                 // (1)
  static template = (model = '') => {                                           // (2)
    return html`<demo-module model=${model}></demo-module>`;
  };
  static duplicateForModelComparison = true;                                    // (3)

  static get styles() {
    return [styles];                                                            // (4)
  }

  private readonly colorService = app.getService(ColorService);                 // (5)

  @observable private pigLatin: string = '';                                    // (6)

  firstUpdated() {
    this.reactImmediately(() => this.selectionService.primarySelectedInputData, // (7)
      primarySelectedInputData => {
        this.getTranslation(primarySelectedInputData);
      });
  }

  private async getTranslation(primarySelectedInputData: IndexedInput) {
    if (primarySelectedInputData === null) return;

    const promise = this.apiService.getPigLatin(primarySelectedInputData);      // (8)
    const results = await this.loadLatest('pigLatin', promise);                 // (9)
    if (results === null) return;

    this.pigLatin = results;
  }

  render() {                                                                    // (10)
    const color = this.colorService.getDatapointColor(
        this.selectionService.primarySelectedInputData);
    return html`
      <div class="results" style=${styleMap({'color': color})}>
        ${this.pigLatin}
      </div>
    `;
  }

  static checkModule(modelSpecs: ModelsMap, datasetSpec: Spec): boolean {       // (11)
    return true;
  }
}

declare global {                                                                // (12)
  interface HTMLElementTagNameMap {
    'demo-module': DemoTextModule;
  }
}
```

The above LitModule, while just a dummy example, illustrates all of the
necessary static properties and many of the most common patterns found in the
LIT app.

### LitModule Setup

First, a `LitModule` must declare a static `title` string (1) and `template`
function (2). The `template` function determines how the modules layout renders
the component template and passes in module properties, such as the name of the
`model` this should respond to. (3) specified behavior in model comparison mode;
if duplicate is set to true, the layout engine will create two (or more)
instances of this module, each responsible for a different model.

_Note: there are additional static attributes which control module behavior; see
the
[`LitModule`](../lit_nlp/client/core/lit_module.ts)
base class for full definitions._

Styles are also declared with a static get method (4), following the lit-element
convention. These styles can be built using the lit-element `css` template
function, or by importing a separate .css file. Styles can be shared between
components by importing a shared styles .css file (for instance,
[`shared_styles.css`](../lit_nlp/client/modules/shared_styles.css))

Services are used by requesting them from the LitApp `app` singleton class (5).
This can be thought of as a super-simple dependency injection system, and allows
for much easier stubbing / mocking of services in testing. We request the
[`colorService`](../lit_nlp/client/services/color_service.ts)
here, but the base `LitModule` class initializes the most common services
([`apiService`](../lit_nlp/client/services/api_service.ts),
[`appState`](../lit_nlp/client/services/state_service.ts),
and
[`selectionService`](../lit_nlp/client/services/selection_service.ts))
for us automatically.

The `LitModule` must also provide a static `checkModule` (11) method, which
determines if this module should display for the given model(s) and dataset.

Finally, the `@customElement('demo-module')` decorator (0) defines this class as
a custom HTML element `<demo-module>`, and (12) ensures this is accessible to
other TypeScript files in different build units.

### LitModule functionality

The above module has a very simple task - When the user selects input data, it
makes a request to an API service to fetch and display a pig latin translation
of the data. Since we're using mobx observables to store and compute our state,
we do this all in a reactive way.

First, since the `LitModule` base class derives from `MobxLitElement`, any
observable data that we use in the `render` method automatically triggers a
rerener when updated. This is excellent for simple use cases, but what about
when we want to trigger more complex behavior, such as the asynchronous request
outlined above?

The pattern that we leverage across the app is as follows: The `render` method
(10) accesses a private observable `pigLatin` property (6) that, when updated,
will rerender the template and show the results of the translation
automatically. In order to update the `pigLatin` observable, we need to set up a
bit of machinery. In the lit-element lifecycle method `firstUpdated`, we use a
helper method `reactImmediately` (7) to set up an explicit reaction to the user
selecting data. Whatever is returned by the first function (in this case
`this.selectionService.primarySelectedInputData`) is observed and passed to the
second function immediately **and** whenever it changes, allowing us to do
something whenever the selection changes. Note, another helper method `react` is
used in the same way as `reactImmediately`, in instances where you don't want to
immediately invoke the reaction.

We pass the selction to the `getTranslation` method to fetch the data from our
API service. However rather than awaiting our API request directly, we pass the
request promise (8) to another helper method `loadLatest` (9). This ensures that
we won't have any race conditions if, for instance, the user selects different
data rapidly - the function returns `null` when the request being fetched has
been superseded by a more recent call to the same endpoint. Finally, we set the
private `pigLatin` observable with the results of our API request and the
template is automatically rerendered, displaying our data.

This may seem like a bit of work for a simple module, but the pattern of using
purely observable data to declaratively specify what gets rendered is very
powerful for simpligying the logic around building larger, more complex
components.

### LitModule Escape Hatches

Finally, it's worth noting that the declarative template-based rendering setup,
while effective for handling most component render logic, is sometimes
inadequate for more advanced visualizations. In particular, the template
approach is not well suited for animations, rapidly changing data, or things
that MUST be done imperatively (such as drawing to a canvas). Fortunately, it's
very easy to "bridge" from declarative to imperative code by leveraging the
lit-element lifecycle methods.

In particular, the `updated` and `firstUpdated` methods are useful for
explicitly doing work after the component has rendered. You can use normal
`querySelector` methods to select elements and update their properties
imperatively (note that you must make selections using the shadow root, not the
document, since we're using isolated web components).

One important caveat is that messing with the actual structure of the rendered
DOM output (such as removing/reordering DOM nodes) **will** cause issues with
lit-element, since it relies on a consistent template output to do its
reconciliation of what needs to be updated per render.

```typescript
// An example of a LITModule imperative "escape hatch"
  updated() {
    const canvas = this.shadowRoot!.querySelector('canvas');
    this.drawCanvas(canvas);
  }

  render() {
    return html`<canvas></canvas>`;
  }
```
