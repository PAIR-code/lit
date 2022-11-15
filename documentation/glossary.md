## Glossary

There are a few commonly-overloaded terms which refer to specific things in the
LIT APIs and codebase:

*   **Component**, a backend component in Python. Includes things like
    counterfactual generators, metrics classes, UMAP and PCA implementations,
    and salience methods.
*   **Element**, a Web Component or another HTML element. The `client/elements/`
    folder contains many custom elements used for visualizations and parts of
    the UI, but which are not full-fledged LIT Modules.
*   **Example** or **Datapoint**, an element of a dataset - the things that we
    feed to models and get predictions back.
*   **Instance**, a specific implementation of LIT (e.g. a demo.py binary) or
    server job running the former.
*   **LIT**, the Learning Interpretability Tool. Always fully capitalized,
    sometimes accompanied by a 🔥 emoji. Pronounced "lit", not "ell-eye-tee".
    Formerly known as the Language Interpretability Tool.
*   **Lit**, the web framework consisting of
    [lit-element](https://lit-element.polymer-project.org/guide) and
    [lit-html](https://lit-html.polymer-project.org/guide) and maintained by the
    Polymer project. LIT is built on this framework, but the naming is
    coincidental (we like it, of course).
*   **Model**, a machine-learning model that we're exploring or debugging with
    LIT. Doesn't need to be a single neural network; could be a pipelined or
    composite system, and may be hosted remotely.
*   **Module**, a frontend visualization module (see
    [Frontend Dev Guide](./frontend_development.md)). Strictly speaking, this is
    something that inherits from LitModule, renders a part of the UI, and
    interacts with the frontend framework. Usually found in `client/modules/`.
    All modules are elements, but not all elements are modules.
*   **Potato** (noun or verb), a frontend error. See
    [potato.io](https://potato.io/).
*   **Server**, the Python backend. A WSGI application that provides a handful
    of HTTP endpoints to serve models, datasets, and other components.
*   **Service**, a part of the frontend framework that handles state and
    provides helper methods. Most of these are global singletons, with the
    notable exception of SelectionService which is duplicated when in
    example-comparison model.
*   **Slice**, a set of examples from a dataset. Often created by **Faceting**
    along a specific feature.
*   **Widget** and **Widget Group**, elements of the frontend layout. A Widget
    is a thin wrapper over a Module, and a Widget Group contains one or more
    widgets along with header bars, resize, and minimize/maximize controls -
    roughly, like a regular GUI window. Sometimes we refer to a widget or a
    widget group as a **Panel**.
