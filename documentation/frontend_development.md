# Frontend Developer Guide

<!--* freshness: { owner: 'lit-dev' reviewed: '2022-02-14' } *-->

<!-- [TOC] placeholder - DO NOT REMOVE -->


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

## Application Architecture

The LIT client frontend is roughly divided into three conceptual groups -
**Modules** (which render visualizations), **Services** (which manage data), and
the **App** itself (which coordinates initialization of services and determines
which modules to render).

### Bootstrapping

The LIT app bootstrapping takes place in two steps: First, the served
[`index.html`](../lit_nlp/client/static/index.html)
page contains a single web component for the
[`<lit-app>`](../lit_nlp/client/core/lit_app.ts).
This component is responsible for the overall layout of the app, including the
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
[`LitApp`](../lit_nlp/client/core/app.ts) singleton
class is provided with a layout declaring which `LitModule` components to use,
then builds the app services and kicks off app initialization and loading data.

### Layout

A layout is defined by a structure of `LitModule` classes, and includes a set of
main components that are always visible, (designated in the object by the "main"
key) and a set of tabs that each contain a group other components.

Layouts are generally specified in Python (see
[Custom Layouts](./api.md#ui-layouts)) through the `LitCanonicalLayout` object.
The default layouts are defined in
[`layout.py`](../lit_nlp/api/layout.py), and you can
add your own by defining one or more `LitCanonicalLayout` objects and passing
them to the server. For an example, see `CUSTOM_LAYOUTS` in
[`lm_demo.py`](../lit_nlp/examples/lm_demo.py).

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
1.  Determining which models/datasets to load and then loading them.

## Modules (LitModule)

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
  static override title = 'Demo Module';                                        // (1)
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>    // (2)
          html`<demo-module model=${model} .shouldReact=${shouldReact}
                selectionServiceIndex=${selectionServiceIndex}></demo-module>`;
  static override duplicateForModelComparison = true;                           // (3)

  static override get styles() {
    return [styles];                                                            // (4)
  }

  private readonly colorService = app.getService(ColorService);                 // (5)

  @observable private pigLatin: string = '';                                    // (6)

  override firstUpdated() {
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

  override renderImpl() {                                                       // (10)
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

### Setup

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
[`shared_styles.css`](../lit_nlp/client/lib/shared_styles.css))

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

### Functionality

The above module has a very simple task - When the user selects input data, it
makes a request to an API service to fetch and display a pig latin translation
of the data. Since we're using mobx observables to store and compute our state,
we do this all in a reactive way.

First, since the `LitModule` base class derives from `MobxLitElement`, any
observable data that we use in the `renderImpl` method automatically triggers a
re-render when updated. This is excellent for simple use cases, but what about
when we want to trigger more complex behavior, such as the asynchronous request
outlined above?

The pattern that we leverage across the app is as follows: The `renderImpl`
method (10) accesses a private observable `pigLatin` property (6) that, when
updated, will re-render the template and show the results of the translation
automatically. In order to update the `pigLatin` observable, we need to set up a
bit of machinery. In the lit-element lifecycle method `firstUpdated`, we use a
helper method `reactImmediately` (7) to set up an explicit reaction to the user
selecting data. Whatever is returned by the first function (in this case
`this.selectionService.primarySelectedInputData`) is observed and passed to the
second function immediately **and** whenever it changes, allowing us to do
something whenever the selection changes. Note, another helper method `react` is
used in the same way as `reactImmediately`, in instances where you don't want to
immediately invoke the reaction. Also note that modules should override
`renderImpl` and not the base `render` method as our `LitModule` base class
overrides `render` with custom logic which calls our `renderImpl` method for
modules to perform their rendering in.

We pass the selection to the `getTranslation` method to fetch the data from our
API service. However rather than awaiting our API request directly, we pass the
request promise (8) to another helper method `loadLatest` (9). This ensures that
we won't have any race conditions if, for instance, the user selects different
data rapidly - the function returns `null` when the request being fetched has
been superseded by a more recent call to the same endpoint. Finally, we set the
private `pigLatin` observable with the results of our API request and the
template is automatically rerendered, displaying our data.

This may seem like a bit of work for a simple module, but the pattern of using
purely observable data to declaratively specify what gets rendered is very
powerful for simplifying the logic around building larger, more complex
components.

### Escape Hatches

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

  override renderImpl() {
    return html`<canvas></canvas>`;
  }
```

### Stateful Child Elements

Some modules may contain stateful child elements, where the element has some
internal state that can have an effect on the module that contains it. Examples of this include any modules that contain the
[elements/faceting_control.ts](../lit_nlp/client/elements/faceting_control.ts) element.

With these types of child elements, it's important for the containing module
to construct them programmatically and store them in a class member variable,
as opposed to only constructing them in the module's html template
string returned by the `renderImpl` method. Otherwise they will be destroyed
and recreated when a module is hidden off-screen and then brought back
on-screen, leading them to lose whatever state they previously held.
Below is a snippet of example code to handle these types of elements.

```typescript
// An example of a LITModule using a stateful child element.
@customElement('example-module')
export class ExampleModule extends LitModule {
  private readonly facetingControl = document.createElement('faceting-control');

  constructor() {
    super();

    const facetsChange = (event: CustomEvent<FacetsChange>) => {
      // Do something with the information from the event.
    };
    // Set the necessary properties on the faceting-control element.
    this.facetingControl.contextName = ExampleModule.title;
    this.facetingControl.addEventListener(
        'facets-change', facetsChange as EventListener)
  }

  override renderImpl() {
    // Render the faceting-control element.
    return html`${this.facetingControl}`;
  }
```
## Style Guide

*   Please disable clang-format on `lit-html` templates and format these
    manually instead:

    ```ts
    // clang-format off
    return html`
      <div class=...>
        <button id=... @click=${doSomething}>Foo</button>
      </div>
    `;
    // clang format on
    ```

*   For new modules, in most cases you should implement two
    classes: one module (subclassing
    [`LitModule`](../lit_nlp/client/core/lit_module.ts))
    that interfaces with the LIT framework, and another element which subclasses
    `LitElement`, `MobxLitElement`, or preferably,
    [`ReactiveElement`](../lit_nlp/client/lib/elements.ts?),
    and implements self-contained visualization code. For an example, see
    [modules/annotated_text_module.ts](../lit_nlp/client/modules/annotated_text_module.ts)
    and
    [elements/annotated_text_vis.ts](../lit_nlp/client/elements/annotated_text_vis.ts).

*   On supported components (`ReactiveElement` and `LitModule`), use
    `this.react()` or `this.reactImmediately()` instead of registering reactions
    directly. This ensures that reactions will be properly cleaned up if the
    element is later removed (such as a layout change or leaving comparison
    mode).

*   Use
    [shared styles](../lit_nlp/client/lib/shared_styles.css)
    when possible.

## Development Tips (open-source)

If you're modifying any TypeScript code, you'll need to re-build the frontend.
You can have yarn do this automatically. In one terminal, run:

```sh
cd ~/lit/lit_nlp
yarn
yarn build --watch
```

And in the second, run the LIT server:

```sh
cd ~/lit
python -m lit_nlp.examples.<example_name> --port=5432 [optional --args]
```

You can then access the LIT UI at http://localhost:5432.

If you only change frontend files, you can use <kbd>Ctrl/Cmd+Shift+R</kbd> to do
a hard refresh in your browser, and it should automatically pick up the updated
source from the build output.

If you're modifying the Python backend, there is experimental support for
hot-reloading the LIT application logic (`app.py`) and some dependencies without
needing to reload models or datasets. See
[`dev_server.py`](../lit_nlp/dev_server.py) for
details.

You can use the `--data_dir` flag (see
[`server_flags.py`](../lit_nlp/server_flags.py) to
save the predictions cache to disk, and automatically reload it on a subsequent
run. In conjunction with `--warm_start`, you can use this to avoid re-running
inference during development - though if you modify the model at all, you should
be sure to remove any stale cache files.

## Custom Client / Modules

The LIT frontend can be extended with custom visualizations or interactive
modules, though this is currently provided as "best effort" support and the API
is not as mature as for Python extensions.

An example of a custom LIT client application, including a custom
(potato-themed) module can be found in
[`lit_nlp/examples/custom_module`](../lit_nlp/examples/custom_module).
You need only define any custom modules (subclass of `LitModule`) and include
them in the build.

When you build the app, specify the directory to build with the `env.build`
flag. For example, to build the `custom_module` demo app:

```sh
yarn build --env.build=examples/custom_module
```

This builds the client app and moves all static assets to a `build` directory in
the specified directory containing the `main.ts` file
(`examples/custom_module/build`).

Finally, to serve the bundle, set the `client_root` flag in your python code to
point to this build directory. For this example we specify the build directory
in `examples/custom_module/potato_demo.py`.

```python
# Here, client build/ is in the same directory as this file
parent_dir = os.path.join(pathlib.Path(__file__).parent.absolute()
FLAGS.set_default("client_root", parent_dir, "build"))
```

You must also define a [custom layout definition](./api.md#ui-layouts) in Python
which references your new module. Note that because Python enums are not
extensible, you need to reference the custom module using its HTML tag name:

```python
modules = layout.LitModuleName
POTATO_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [modules.DatapointEditorModule, modules.ClassificationModule],
    },
    lower={
        "Data": [modules.DataTableModule, "potato-module"],
    },
    description="Custom layout with our spud-tastic potato module.",
)
```

See
[`potato_demo.py`](../lit_nlp/examples/custom_module/potato_demo.py)
for the full example.
