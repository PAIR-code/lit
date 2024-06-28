# LIT Python API

<!--* freshness: { owner: 'lit-dev' reviewed: '2023-08-23' } *-->

<!-- [TOC] placeholder - DO NOT REMOVE -->

## Design Overview

LIT is a modular system, comprising a collection of backend components (written
in Python) and frontend modules (written in TypeScript). Most users will develop
against the Python API, which is documented below and allows LIT to be extended
with custom models, datasets, metrics, counterfactual generators, and more. The
LIT server and components are provided as a library which users can use through
their own demo binaries or via Colab.

The components can also be used as regular Python classes without starting a
server; see [below](#using-lit-components-outside-of-lit) for details.

![LIT system overview](./images/lit-system-diagram.svg)

The LIT backend serves models, data, and interpretability components, each of
which is a Python class implementing a minimal API and relying on the
[spec system](#type-system) to detect fields and verify compatibility. The
server is stateless, but implements a caching layer for model predictions - this
simplifies component design and allows interactive use of large models like BERT
or T5.

The frontend is a stateful single-page app, built using
[Lit](https://lit.dev/)[^1] for modularity and [MobX](https://mobx.js.org/) for
state management. It consists of a core UI framework, a set of shared "services"
which manage persistent state, and a set of independent modules which render
visualizations and support user interaction. For more details, see the
[UI guide](./ui_guide.md) and the
[frontend developer guide](./frontend_development.md).

[^1]: Naming is just a happy coincidence; the Learning Interpretability Tool is
      not related to the Lit projects.


## Adding Models and Data

To run LIT with your own models and data, you can create a custom `demo.py`
script that passes these to the LIT server. For example:

```py
def main(_):
  # MulitiNLIData implements the Dataset API
  datasets = {
      'mnli_matched': MultiNLIData('/path/to/dev_matched.tsv'),
      'mnli_mismatched': MultiNLIData('/path/to/dev_mismatched.tsv'),
  }

  # NLIModel implements the Model API
  models = {
      'model_foo': NLIModel('/path/to/model/foo/files'),
      'model_bar': NLIModel('/path/to/model/bar/files'),
  }

  lit_demo = lit_nlp.dev_server.Server(models, datasets, port=4321)
  lit_demo.serve()

if __name__ == '__main__':
  main()
```

Conceptually, a dataset is just a list of examples and a model is just a
function that takes examples and returns predictions. The [`Dataset`](#datasets)
and [`Model`](#models) classes implement this, and provide metadata (see the
[type system](#type-system)) to describe themselves to other components.

For pre-built `demo.py` examples, check out
https://github.com/PAIR-code/lit/tree/main/lit_nlp/examples

### Validating Models and Data

Datasets and models can optionally be validated by LIT to ensure that dataset
examples match their spec and that model output values match their spec.
This can be very helpful during development of new model and dataset wrappers
to ensure correct behavior in LIT.

At LIT server startup, the `validate` flag can be used to enable validation.
There are three modes:

*   `--validate=first` will check the first example in each dataset.
*   `--validate=sample` will validate a sample of 5% of each dataset.
*   `--validate=all` will run validation on all examples from all datasets.

Additionally, if using LIT datasets and models outside of the LIT server,
validation can be called directly through the
[`validation`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/lib/validation.py) module.

## Datasets

Datasets ([`Dataset`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/dataset.py)) are
just a list of examples, with associated type information following LIT's
[type system](#type-system).

*   `spec()` should return a flat dict that describes the fields in each example
*   `self._examples` should be a list of flat dicts

LIT operates on all examples loaded in the datasets you include in your LIT
server, therefore you should take care to use dataset sizes that can fit into
memory on your backend server and can be displayed in the browser.

NOTE: See the [FAQ](./faq.md) for more details on dataset size limitations.

Implementations should subclass
[`Dataset`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/dataset.py). Usually this
is just a few lines of code - for example, the following is a complete
implementation for the [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)
dataset:

```py
class MultiNLIData(Dataset):
  """Loader for MultiNLI development set."""

  NLI_LABELS = ['entailment', 'neutral', 'contradiction']

  def __init__(self, path: str):
    # Read the eval set from a .tsv file as distributed with the GLUE benchmark.
    df = pandas.read_csv(path, sep='\t')
    # Store as a list of dicts, conforming to self.spec()
    self._examples = [{
      'premise': row['sentence1'],
      'hypothesis': row['sentence2'],
      'label': row['gold_label'],
      'genre': row['genre'],
    } for _, row in df.iterrows()]

  def spec(self) -> types.Spec:
    return {
      'premise': lit_types.TextSegment(),
      'hypothesis': lit_types.TextSegment(),
      'label': lit_types.CategoryLabel(vocab=self.NLI_LABELS),
      # We can include additional fields, which don't have to be used by the model.
      'genre': lit_types.CategoryLabel(),
    }
```

In this example, all four fields (premise, hypothesis, label, and genre) have
string values, but the [semantic types](#type-system) tell LIT a bit more about
how to interpret them:

*   `premise` and `hypothesis` should be treated as natural-language text
    (`TextSegment`)
*   `label` should be treated as a categorical feature (`CategoryLabel`) with a
    fixed, known set of possible values (`vocab=self.NLI_LABELS`)
*   `genre` should be treated as a categorical feature, but with an unknown or
    open set of values.

This implementation uses Pandas to read a TSV file, but you can also use
services like [TensorFlow Datasets](https://www.tensorflow.org/datasets) -
simply wrap them in your `__init__()` function.

Note that you can freely add additional features - such as `genre` in the
example above - which the model may not be aware of. The LIT UI will recognize
these features for slicing, binning, etc., and they will also be available to
interpretation components such as custom metrics.

### Transformations

The `Dataset` class also supports a limited set of transformations, similar to
TensorFlow's
[tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
but more limited in scope and aimed at supporting quick iteration:

*   `Dataset.slice[start:step:end]` will return a new `Dataset` with the same
    spec and a slice of the datapoints.
*   `Dataset.sample(n, seed=42)` will return a new `Dataset` with the same spec
    and a random sample of the datapoints.
*   `Dataset.remap(field_map: dict[str, str])` will return a new `Dataset` with
    renamed fields in both the examples and spec.

The latter is a shortcut to use datasets matching one model with another; for
example, a dataset with a `"document"` field can be used with a model expecting
a `"text"` input via `Dataset.remap({"document":
"text"})`.[^why-not-standardize-names]

[^why-not-standardize-names]: We could solve this particular case by
    standardizing names, but one still needs to be
    explicit if there are multiple segments available,
    such as `"question"` and `"document"` for a QA
    task.

## Models

Models ([`Model`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/model.py)) are
functions which take inputs and produce outputs, with associated type
information following LIT's [type system](#type-system). The core API consists
of three methods:

*   `input_spec()` should return a flat dict that describes necessary input
    fields
*   `output_spec()` should return a flat dict that describes the model's
    predictions and any additional outputs
*   `predict()` should take a sequence of inputs (satisfying `input_spec()`) and
    yields a parallel sequence of outputs matching `output_spec()`.

Implementations should subclass
[`Model`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/model.py). An example for
[MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) might look something like:

```py
class NLIModel(Model):
  """Wrapper for a Natural Language Inference model."""

  NLI_LABELS = ['entailment', 'neutral', 'contradiction']

  def __init__(self, model_path: str, **kw):
    # Load the model into memory so we're ready for interactive use.
    self._model = _load_my_model(model_path, **kw)

  ##
  # LIT API implementations
  def predict(self, inputs: Iterable[Input]) -> Iterable[Preds]:
    """Predict on a stream of examples."""
    examples = [self._model.convert_dict_input(d) for d in inputs]  # any custom preprocessing
    return self._model.predict_examples(examples)  # returns a dict for each input

  def input_spec(self) -> types.Spec:
    """Describe the inputs to the model."""
    return {
        'premise': lit_types.TextSegment(),
        'hypothesis': lit_types.TextSegment(),
    }

  def output_spec(self) -> types.Spec:
    """Describe the model outputs."""
    return {
      # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
      'probas': lit_types.MulticlassPreds(vocab=NLI_LABELS, parent='label'),
    }
```

Unlike the dataset example, this model implementation is incomplete - you'll
need to customize `predict()` accordingly with any pre- or post-processing
needed, such as tokenization.

Many deep learning models support a batched prediction behavior. Thus, we
provide the `BatchedModel` class that implements simple batching. Users of this
class must implement the `predict_minibatch()` function, which should convert
a `Sequence` of `JsonDict` objects to the appropriate batch representation
(typically, a `Mapping` of strings to aligned `Sequences` or `Tensors`) before
calling the model. Optionally, you may want to override the
`max_minibatch_size()` function, which determines the batch size.

Note: there are a few additional methods in the model API - see
[`Model`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/model.py) for details.

If your model is on a remote server, consider using the `BatchedRemoteModel`
base class, which implements parallel batched requests using a thread pool.

### Adding more outputs

The above example defined a black-box model, with predictions but no access to
internals. If we want a richer view into the model's behavior, we can add
additional return fields corresponding to hidden-state activations, gradients,
word embeddings, attention, or more. For example, a BERT-based model with
several such features might have the following `output_spec()`:

```py
  def output_spec(self) -> types.Spec:
    """Describe the model outputs."""
    return {
      # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
      'probas': lit_types.MulticlassPreds(vocab=NLI_LABELS, parent='label'),
      # This model returns two different embeddings (activation vectors), but you can easily add more.
      'output_embs': lit_types.Embeddings(),      # from [CLS] token at top layer
      'mean_word_embs':  lit_types.Embeddings(),  # mean of input word embeddings
      # In LIT, we treat tokens as another model output. There can be more than one,
      # and the 'parent' field describes which input segment they correspond to.
      'premise_tokens': lit_types.Tokens(parent='premise'),
      'hypothesis_tokens': lit_types.Tokens(parent='hypothesis'),
      # Gradients are also returned by the model; 'align' here references a Tokens field.
      'premise_grad': lit_types.TokenGradients(align='premise_tokens'),
      'hypothesis_grad': lit_types.TokenGradients(align='hypothesis_tokens'),
      # Similarly, attention references a token field, but here we want the model's full "internal"
      # tokenization, which might be something like: [START] foo bar baz [SEP] spam eggs [END]
      'tokens': lit_types.Tokens(),
      'attention_layer0': lit_types.AttentionHeads(align=['tokens', 'tokens']),
      'attention_layer1': lit_types.AttentionHeads(align=['tokens', 'tokens']),
      'attention_layer2': lit_types.AttentionHeads(align=['tokens', 'tokens']),
      # ...and so on. Since the spec is just a dictionary of dataclasses, you can populate it
      # in a loop if you have many similar fields.
    }
```

The `predict()` function would return, for each example, additional dict entries
corresponding to each of these fields.

Note: Because tokenization is often tightly coupled with the model code, we
treat it as an intermediate state on the same level as embeddings or attention,
and thus return `Tokens` as a field in the model *output*. This also allows
models to expose different tokenizations for different inputs, such as
`premise_tokens` and `hypothesis_tokens` above.

LIT components and frontend modules will automatically detect these spec fields
and use them to support additional interpretation methods, such as the embedding
projector or gradient-based salience maps.

You can also implement multi-headed models this way: simply add additional
output fields for each prediction (such as another `MulticlassPreds`), and
they'll be automatically detected.

See the [type system documentation](#type-system) for more details on available
types and their semantics.

### Optional inputs

By default, LIT treats `input_spec` fields as required. However, this can be set
to false if you wish to define optional model inputs. For example, a model that
can accept pre-tokenized inputs might have the following spec:

```python
    def input_spec(self) -> types.Spec:
      return {
          "text": lit_types.TextSegment(),
          "tokens": lit_types.Tokens(parent='text', required=False),
      }
```

And in the model's `predict()`, you would have logic to use these and bypass the
tokenizer:

```python
    def predict(self, inputs: Iterable[Input]) -> Iterable[Preds]:
      input_tokens = [ex.get('tokens') or self.tokenizer.tokenize(ex['text'])
                      for ex in inputs]
      # ...rest of your predict logic...
```

`required=False` can also be used for label fields (such as `"label":
lit_types.CategoryLabel(required=False)`), though these can also be omitted from
the input spec entirely if they are not needed to compute model outputs.

## Interpretation Components

Backend interpretation components include metrics, salience maps, visualization
aids like [UMAP](https://umap-learn.readthedocs.io/en/latest/), and
counterfactual generator plug-ins.

Most such components implement the
[`Interpreter`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/components.py) API.
Conceptually, this is any function that takes a set of datapoints and a model,
and produces some output.[^identity-component] For example,
[local gradient-based salience (GradientNorm)](https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py)
processes the `TokenGradients` and `Tokens` returned by a model and produces a
list of scores for each token. The Integrated Gradients saliency method
additionally requires a `TokenEmbeddings` input and corresponding output, as
well as a label field `Target` to pin the gradient target to the same class as
an input and corresponding output. See the
[GLUE models class](https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/models/glue_models.py)
for an example of these spec requirements.

The core API involves implementing the `run()` method:

```python
  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    # config is any runtime options to this component, such as a threshold for
    # (binary) classification metrics.
```

Output from an interpreter component is unconstrained; it's up to the frontend
component requesting it to process the output correctly. In particular, some
components (such as salience maps) may operate on each example independently,
similar to model predictions, while others (such as metrics) may produce
aggregate summaries of the input set.

Interpreters are also responsible for verifying compatibility by reading the
model and dataset specs; these are also used to determine what fields to operate
on. A typical implementation just loops over the relevant specs. For example,
for
[simple gradient-based salience](https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py)
we might have:

```python
  def find_fields(self, output_spec: Spec) -> list[str]:
    # Find TokenGradients fields
    grad_fields = utils.find_spec_keys(output_spec, types.TokenGradients)

    # Check that these are aligned to Tokens fields
    for f in grad_fields:
      tokens_field = output_spec[f].align  # pytype: disable=attribute-error
      assert tokens_field in output_spec
      assert isinstance(output_spec[tokens_field], types.Tokens)
    return grad_fields

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[list[JsonDict]]:
    """Run this component, given a model and input(s)."""
    # Find gradient fields to interpret
    output_spec = model.output_spec()
    grad_fields = self.find_fields(output_spec)
    logging.info('Found fields for gradient attribution: %s', str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      return None

    # do rest of the work to create the salience maps for each available field

    # return a dtypes.TokenSalience for each input, which has a list of
    # tokens (from the model) and their associated scores.
```

This design adds some code overhead to interpretation components, but the
benefit is flexibility - Python can be used to specify complex dependencies
between fields, and multiple outputs can be easily supported in a loop.

[^identity-component]: A trivial one might just run the model and return
    predictions, though in practice we have a separate
    endpoint for that.

### Metrics

For metrics, the
[`SimpleMetrics`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/metrics.py)
class implements the spec-matching and input-unpacking logic to satisfy the
general `Interpreter` API. A subclass of `SimpleMetrics` should implement an
`is_compatible()` method and a `compute()` method, which is called on compatible
(prediction, label) pairs and returns a dict of named score fields. For example:

```python
class RegressionMetrics(SimpleMetrics):
  """Standard regression metrics."""

  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if compatible with this field."""
    return isinstance(field_spec, types.RegressionScore)

  def compute(self,
              labels: Sequence[float],
              preds: Sequence[float],
              label_spec: types.Scalar,
              pred_spec: types.RegressionScore,
              config: Optional[JsonDict] = None) -> dict[str, float]:
    """Compute metric(s) between labels and predictions."""
    del config
    mse = sklearn_metrics.mean_squared_error(labels, preds)
    pearsonr = scipy_stats.pearsonr(labels, preds)[0]
    spearmanr = scipy_stats.spearmanr(labels, preds)[0]
    return {'mse': mse, 'pearsonr': pearsonr, 'spearmanr': spearmanr}
```

The implementation of `SimpleMetrics.run()` uses the `parent` key (see
[type system](#type-system)) in fields of the model's output spec to find the
appropriate input fields to compare against, and calls `compute()` accordingly
on the unpacked values.

### Generators

Conceptually, a generator is just an interpreter that returns new input
examples. These may depend on the input only, as for techniques such as back-
translation, or can involve feedback from the model, such as for adversarial
attacks.

The core generator API is:

```python
class Generator(Interpreter):
  """Base class for LIT generators."""

  def generate_all(self,
                   inputs: list[JsonDict],
                   model: lit_model.Model,
                   dataset: lit_dataset.Dataset,
                   config: Optional[JsonDict] = None) -> list[list[JsonDict]]:
    """Run generation on a set of inputs.

    Args:
      inputs: sequence of inputs, following model.input_spec()
      model: optional model to use to generate new examples.
      dataset: optional dataset which the current examples belong to.
      config: optional runtime config.

    Returns:
      list of list of new generated inputs, following model.input_spec()
    """
```

Where the output is a list of lists: a set of generated examples for each input.
For convenience, there is also a `generate()` method which takes a single
example and returns a single list; we provide the more general `generate_all()`
API to support model-based generators (such as back-translation) which benefit
from batched requests.

As with other interpreter components, a generator can take custom arguments
through `config`, such as the list of substitutions for the
[word replacer](https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/word_replacer.py).

#### Backtranslator Generator

The [backtranslator](https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/backtranslator.py)
generator translates text segment inputs into foreign languages and back to the
source language in order to create paraphrases.
It relies on the Google Cloud Translate API to perform those translations.
To use it, you must have a Google Cloud project and set up Cloud Translation
as described at https://cloud.google.com/translate/docs/setup.
Then, download  your application credentials file locally and set the
GOOGLE_APPLICATION_CREDENTIALS environment variable to point to that file.
With that environment variable set to the correct path, LIT can make use of the
backtranlator generator if you pass it as a generator in the Server constructor.

### Configuration UI

Interpreter components support an optional `config` option to specify run-time
options, such as the number of samples for LIME or the pivot languages for
back-translation. LIT provides a simple DSL to define these options, which will
auto-generate a form on the frontend. The DSL uses the same
[type system](#type-system) as used to define data and model outputs, and the
`config` argument will be passed a dict with the form values.

For example, the following spec:

```python
  def config_spec(self) -> types.Spec:
    return {
        "Pivot languages": types.SparseMultilabel(
            vocab=['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw',
                   'th', 'tr', 'ur', 'vi', 'zh'],
            default=['de', 'fr']),
        "Source language": types.TextSegment(default='en'),
    }
```

will give this form to configure back-translation:

![Back-translation Config Form](./images/api/backtranslation-form-example.png){w=400px align=center}

Currently `config_spec()` is supported only for generators and salience methods,
though any component can support the `config` argument to its `run()` method,
which can be useful if
[running outside of the LIT UI](#using-lit-components-outside-of-lit).

The following [types](#available-types) are supported (see
[interpreter_controls.ts](https://github.com/PAIR-code/lit/blob/main/lit_nlp/client/elements/interpreter_controls.ts)):

*   `Scalar`, which creates a slider for setting a numeric option. You can
    specify the `min_val`, `max_val`, `default`, and `step`, values for the
    slider through arguments to the `Scalar` constructor.
*   `Boolean` (`BooleanLitType` in TypeScript), which creates a checkbox, with
    a `default` value to be set in the constructor.
*   `CategoryLabel`, which creates a dropdown with options specified in the
    `vocab` argument.
*   `SparseMultilabel`, which creates a series of checkboxes for each option
    specified in the `vocab` argument.
*   `TextSegment`, which creates an input text box for string entry, with an
    optional default value from the `default` argument.
*   `Tokens`, which creates an input text box for entry of multiple,
    comma-separated strings which are parsed into a list of strings to be
    supplied to the interpreter.
*   `SingleFieldMatcher`, which acts like a `CategoryLabel` but where the vocab
    is automatically populated by the names of fields from the data or model
    spec. For example, `SingleFieldMatcher(spec='dataset',
    types=['TextSegment'])` will give a dropdown with the names of all
    `TextSegment` fields in the dataset.
*   `MultiFieldMatcher` is similar to `SingleFieldMatcher` except it gives a set
    of checkboxes to select one or more matching field names. The returned value
    in `config` will be a list of string values.

The field matching controls can be useful for selecting one or more fields to
operate on. For example,to choose which input fields to perturb, or which output
field of a multi-head model to run an adversarial attack (such as HotFlip)
against.

## Type System

LIT passes data around (e.g., between the server and the web app) as flat
records with `string` keys. In Python types these are `Mapping[str, ...]` and in
TypeScript types these are `{[key: string]: unknown}`. LIT serializes these
records to JSON when communicating between the server and the web app client. It
is because of this serialization that we introduced LIT's type system; LIT needs
a way to communicate how to process and understand the semantics of the _shape_
and (allowable) _values_ for the records being passed around in
[JSON's more limited type system][json].

<!--
  TODO(b/290782213): Update the serialization discussion above to reflect any
  changes to LIT's wrire format for HTTP APIs.
-->

<!--
  TODO(b/258531316): Update Spec type once converted to a readonly type
-->
The _shape_ of a record &ndash; its specific keys and the types of their values
&ndash; is defined by a `Spec`; a `dict[str, LitType]`. Each `LitType` class has
a `default` property whose type annotation describes the type of the _value_ for
that field in a JSON record. `LitType`s are implemented using hierarchical
inheritance; the canonical types can be found in [types.py][types_py], with
parallel implementations in [lit_types.ts][types_ts].

### Conventions

LIT supports several different "kinds" of `Spec`s (input vs output vs meta,
etc.), and their use in context has specific implications, described
per base class below.

* [`lit_nlp.api.dataset.Dataset`][dataset-py]
    * **`.spec() -> Spec`** describes the shape of every record in the
      `Sequence` returned by `Dataset.examples()`.
    * **`.init_spec() -> Optional[Spec]`** describes the user-configurable
      arguments for loading a new instance of this `Dataset` class via the web
      app's UI. Returning `None` or an empty `Spec` means that there is nothing
      configurable about how this `Dataset` is loaded, and it will not show up
      in the dataset loading section of the web app's Global Settings.
* [`lit_nlp.api.model.Model`][model-py]
    * **`.input_spec() -> Spec`** describes the shape required of all records
      passed into the `Model.predict()` function via the `inputs` argument. LIT
      checks for compatibility between a `Dataset` and a `Model` by ensuring
      that `Model.input_spec()` is a subset of `Dataset.spec()`.
    * **`.output_spec() -> Spec`** describes the shape of all records returned
      by the `Model.predict()` function.
    * **`.init_spec() -> Optional[Spec]`** describes the user-configurable
      arguments for loading a new instance of this `Model` class via the web
      app's UI. Returning `None` or an empty `Spec` means that there is nothing
      configurable about how this `model` is loaded, and it will not show up in
      the model loading section of the web app's Global Settings.
* [`lit_nlp.api.components.[Interpreter | Generator]`][components-py]
    * **`.config_spec() -> Spec`** describes the user-configurable parameters
      for running this component. Returning an empty `Spec` means that this
      component always processes inputs in the same way.
    * **`.meta_spec() -> Spec`** is essentially unconstrained, but ideally
      describes the shape of the records returned by this component's `.run()`
      method. Note that this `Spec` has different semantics depending on the
      component type. `Interpreter.run()` typically returns an
      `Iterable[Mapping[str, ...]]` of records (i.e., the `Mapping`) with this
      shape, because each input corresponds to one interpretation. Whereas
      `Generator.run()` typically returns an
      `Iterable[Iterable[Mapping[str, ...]]]` of records with this shape,
      because each input may enable the generation of one or more new examples.
* [`lit_nlp.api.components.Metrics`][components-py]
    * **`.config_spec() -> Spec`** describes the user-configurable parameters
      for running this component. Returning an empty `Spec` means that this
      component always processes inputs in the same way.
    * **`.meta_spec() -> Spec`** is a slight variation on the tradition `Spec`;
      it will always be a `Mapping[str, MetricResult]` describing the single
      record returned by the `Metrics.run()` method for each pair of compatible
      keys in the `Model.output_spec()` and `Dataset.spec()`. The `MetricResult`
      type also describes how to interpret the values in each record, e.g., if
      higher, lower, or numbers closer to zero are better.

Each `LitType` subclass encapsulates its own semantics (see
[types.py][types_py]), but there are a few conventions all subclasses follow:

*   The **`align=` attribute** references another field _in the same spec_ and
    implies that both fields have index-aligned elements. For
    example, `Model.output_spec()` may contain `'tokens': lit_types.Tokens(...)`
    and `'pos': lit_types.SequenceTags(align='tokens')`, which references the
    "tokens" field. This implies that the "pos" field contains a corresponding
    value for every item in "tokens" and that you can access them with numeric
    indices. Transitively, this means that using `zip(..., strict=True)` (in
    Python 3.10 and above) will act as a pseudo-validator of this expectation.

*   The **`parent=` attribute** is _typically_ used by `LitType`s in a
    `Model.output_spec()`, and must be a field in the _input spec_ (i.e. the
    `Dataset.spec()`) against which this field's value will be compared. For
    example, the `Model.output_spec()` may contain
    `'probas': lit_types.MulticlassPreds(parent='label', ...)` and the
    `Dataset.spec()` may contain `'label': lit_types.CategoryLabel()`, which
    means that the `Dataset`'s "label" field contains the ground truth values
    for that example, and the class prediction in the "probas" field can be
    compared to this label, e.g., by multi-class metrics.

*   The **`vocab=` attribute** is used to represent the allowable values for
    that field, such as a set of classes for a `MulticlassPreds` field, or the
    set of labels for a `CategoryLabel` field.

*   A field that appears in _both_ the model's input and output specs is assumed
    to represent the same value. This pattern is used for model-based input
    manipulation. For example, a
    [language model](https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/models/pretrained_lms.py)
    might output `'tokens': lit_types.Tokens(...)`, and accept as (optional)
    input `'tokens': lit_types.Tokens(required=False, ...)`. An interpretability
    component could take output from the former, swap one or more tokens (e.g.
    with `[MASK]`), and feed them in the corresponding input field to compute
    masked fills.

### Compatibility Checks

LIT's type system plays a critical role in ensuring reliability of and
interoperability between the `Model`, `Dataset`, `Interpreter`, `Generator`, and
`Metrics` classes:

*   The **Model-Dataset compatibility check** ensures that the
    `Model.input_spec()` is a subset of the `Dataset.spec()`. The base
    [`Model` class][model-py] provides a robust and universal implementation of
    this check in the `is_compatible_with_dataset()` API, but you can override
    this method in your `Model` subclass if you so choose.
*   All [`lit_nlp.api.components` classes][components-py] provide an
    `is_compatible` API to check their compatibility against `Model`s and
    `Dataset`s, as appropriate. For example, the
    [`WordReplacer` generator][word-replacer] only checks against the `Dataset`
    spec because it does not depend on model outputs, whereas the
    [`Curves` interpreter][curves-interp] checks the `Model` and `Dataset`
    because it needs labeled predictions, and the
    [`GradientDotInput` interpreter][grad-maps] only checks against the
    `Model.output_spec()` because it needs data that only the model can provide.

The LIT web app also uses `Spec` based compatibility checks. Each TypeScript
module defines a [`shouldDisplayModule` function][should_display_module] that
returns `true` if any active model-dataset pair provides sufficient information
to support the visualization methods encapsulated by that module. If this
function returns `false`, the module is not displayed in the layout. Note that
this can cause jitter (UI modules appearing, disappearing, reordering, resizing,
etc.) when switching between models or datasets with heterogeneous `Spec`s.

When implementing your own LIT components and modules, you can use
[`utils.find_spec_keys()`][utils-lib]
(Python) and
[`findSpecKeys()`][utils-lib]
(TypeScript) to identify fields of interest in a `Spec`. These methods recognize
and respect subclasses. For example,
`utils.find_spec_keys(spec, Scalar)` will also match any `RegressionScore`
fields, but `utils.find_spec_keys(spec, RegressionScore)` will not return all
`Scalar` fields in the `Spec`.

Important: Compatibility checks are performed automatically when
[building the `LitMetadata`][build-metadata] for an instance of `LitApp`,
typically by calling `dev_server.Serve()`. **These checks are not performed when
using components in a raw Python context** (e.g., Colab, Jupyter, a REPL), as
[described below](#using-lit-components-outside-of-lit), and it is encouraged
that you call these explicitly to ensure compatibility and avoid chasing red
herrings.

### An In-Depth Example

Consider the following example from the [MNLI demo][mnli-demo]. The
[MultiNLI][mnli-dataset] dataset might define the following `Spec`.

```python
# Dataset.spec()
{
  "premise": lit_types.TextSegment(),
  "hypothesis": lit_types.TextSegment(),
  "label": lit_types.CategoryLabel(
      vocab=["entailment", "neutral", "contradiction"]
  ),
  "genre": lit_types.CategoryLabel(),
}
```

An example record in this `Dataset` might be:

```python
# dataset.examples[0]
{
  "premise": "Buffet and a la carte available.",
  "hypothesis": "It has a buffet."
  "label": "entailment",
  "genre": "travel",
}
```

A classification model for this task might have the following `input_spec()` and
`output_spec()`. Notice that the input spec is a subset of the `Dataset.spec()`,
thus LIT considers these to be compatible.

```python
# model.input_spec()
{
  "premise": lit_types.TextSegment(),
  "hypothesis": lit_types.TextSegment(),
}

# model.output_spec()
{
  "probas": lit_types.MulticlassPreds(
      parent="label",
      vocab=["entailment", "neutral", "contradiction"]
  ),
}
```

Running this model over the input might yield the following prediction.

```python
# model.predict([dataset.examples[0]])[0]
{
  "probas": [0.967, 0.024, 0.009],
}
```

Passing this input and the prediction to the `ClassificationResults` interpreter
would yield additional human-readable information as follows.

```python
# classification_results.run(
#     dataset.examples[:1], model, dataset, [prediction]
# )[0]
{
  "probas": {
      "scores": [0.967, 0.024, 0.009],
      "predicted_class": "entailment",
      "correct": True,
  },
}
```

_See the [examples](https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples) for more._

### Available types

The full set of `LitType`s is defined in [types.py](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/types.py). Numeric types such as `Integer` and `Scalar` have predefined ranges that can be overridden using corresponding `min_val` and `max_val` attributes as seen [here](https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/datasets/penguin_data.py;l=19-22;rcl=574999438). The different types available in LIT are summarized
in the table below.

Note: Bracket syntax, such as `<float>[num_tokens]`, refers to the shapes of
NumPy arrays where each element inside the brackets is an integer.

Name                      | Description                                                                                                                                                           | Value Type
------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------
`TextSegment`             | Natural language text, untokenized.                                                                                                                                   | `str`
`GeneratedText`           | Untokenized text, generated from a model (such as seq2seq).                                                                                                           | `str`
`URL`                     | TextSegment, but interpreted as a URL.                                                                                                                                | `str`
`GeneratedURL`            | Generated TextSegment, but interpreted as a URL (i.e., it maye not be real/is inappropriate as a label).                                                              | `str`
`SearchQuery`             | TextSegment, but interpreted as a search query.                                                                                                                       | `str`
`String`                  | Opaque string data; ignored by components such as perturbation methods that operate on natural language.                                                              | `str`
`ReferenceTexts`          | Multiple texts, such as a set of references for summarization or MT.                                                                                                  | `list[tuple[str, float]]`
`GeneratedTextCandidates` | Multiple generation candidates, such as beam search output from a seq2seq model.                                                                                      | `list[tuple[str, float]]`
`Tokens`                  | Tokenized text.                                                                                                                                                       | `list[str]`
`TokenTopKPreds`          | Predicted tokens and their scores, as from a language model or seq2seq model.                                                                                         | `list[list[tuple[str, float]]]`
`Boolean`                 | Boolean value.                                                                                                                                                        | `bool`
`Scalar`                  | Scalar numeric value.                                                                                                                                                 | `float`
`Integer`                 | Integer, with a default range from -32768 to +32767. value.                                                                                                                                                        | `int`
`ImageBytes`              | Image, represented by a base64 encoded string. LIT also provides `JPEGBytes` and `PNGBytes` types for those specific encodings.                                       | `str`
`RegressionScore`         | Scalar value, treated as a regression target or prediction.                                                                                                           | `float`
`ReferenceScores`         | Scores for one or more reference texts.                                                                                                                               | `list[float]`
`CategoryLabel`           | Categorical label, from open or fixed vocabulary.                                                                                                                     | `str`
`MulticlassPreds`         | Multiclass predicted probabilities.                                                                                                                                   | `<float>[num_labels]`
`SparseMultilabel`        | Multiple non-exclusive labels, such as a set of attributes.                                                                                                           | `list[str]`
`SparseMultilabelPreds`   | Sparse multi-label predictions, represented as scored candidates.                                                                                                     | `list[tuple[str, float]]`
`SequenceTags`            | Sequence tags, aligned to tokens.                                                                                                                                     | `list[str]`
`SpanLabels`              | Span labels, aligned to tokens. Each label is (i,j,label).                                                                                                            | `list[SpanLabel]`
`EdgeLabels`              | Edge labels, aligned to tokens. This is a general way to represent many structured prediction tasks, such as coreference or SRL. See https://arxiv.org/abs/1905.06316 | `list[EdgeLabel]`
`MultiSegmentAnnotations` | In-line byte-span annotations, which can span multiple text segments.                                                                                                 | `list[AnnotationCluster]`
`Embeddings`              | Fixed-length embeddings or model activations.                                                                                                                         | `<float>[emb_dim]`
`Gradients`               | Gradients with respect to embeddings or model activations.                                                                                                            | `<float>[emb_dim]`
`TokenEmbeddings`         | Per-token embeddings or model activations.                                                                                                                            | `<float>[num_tokens, emb_dim]`
`TokenGradients`          | Gradients with respect to per-token embeddings or model activations.                                                                                                  | `<float>[num_tokens, emb_dim]`
`ImageGradients`          | Gradients with respect to image pixels.                                                                                                                               | `<float>[image_height, image_width, color_channels]`
`AttentionHeads`          | Attention heads, grouped by layer.                                                                                                                                    | `<float>[num_heads, num_tokens, num_tokens]`

Values can be plain data, NumPy arrays, or custom dataclasses - see
[dtypes.py](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/dtypes.py) for further
detail.

*Note: Note that `String`, `Boolean` and `URL` types in Python are represented
as `StringLitType`, `BooleanLitType` and `URLLitType` in TypeScript to avoid
naming collisions with protected TypeScript keywords.*

## Server Configuration

Some properties of the LIT frontend can be configured from Python as
**arguments to `dev_server.Server()`**. These include:

*   `page_title`: set a custom page title, such as "Coreference Demo".
*   `canonical_url`: set a "canonical" URL (such as a shortlink) that will be
    used as the base when copying links from the LIT UI.
*   `default_layout`: set the default UI layout, by name. See `layout.ts` and
    the section below for available layouts.
*   `demo_mode`: demo / kiosk mode, which disables some functionality (such as
    save/load datapoints) which you may not want to expose to untrusted users.
*   `inline_doc`: a markdown string that will be rendered in a documentation
    module in the main LIT panel.
*   `onboard_start_doc`: a markdown string that will be rendered as the first
    panel of the LIT onboarding splash-screen.
*   `onboard_end_doc`: a markdown string that will be rendered as the last
    panel of the LIT onboarding splash-screen.

For detailed documentation, see
[server_flags.py](https://github.com/PAIR-code/lit/blob/main/lit_nlp/server_flags.py).

Most Python components (such as `Model`, `Dataset`, and `Interpreter`) also have
a `description()` method which can be used to specify a human-readable
description or help text that will appear in the UI.

### Customizing the Layout

You can specify custom web app layouts from Python via the `layouts=` attribute.
The value should be a `Mapping[str, LitCanonicalLayout]`, such as:

```python
LM_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [
            modules.EmbeddingsModule,
            modules.DataTableModule,
            modules.DatapointEditorModule,
        ]
    },
    lower={
        "Predictions": [
            modules.LanguageModelPredictionModule,
            modules.ConfusionMatrixModule,
        ],
        "Counterfactuals": [modules.GeneratorModule],
    },
    description="Custom layout for language models.",
)
```

You can pass this to the server as:

```python
lit_demo = dev_server.Server(
    models,
    datasets,
    # other args...
    layouts={"lm": LM_LAYOUT},
    **server_flags.get_flags())
return lit_demo.serve()
```

For a full example, see
[`lm_demo.py`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/lm_demo.py).

You can see the pre-configured layouts provided by LIT, as well as the list of
modules that can be included in your custom layout in
[`layout.py`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/layout.py). A
`LitCanonicalLayout` can be defined to achieve four different configurations of
the major content areas:

* Single-panel: Define only the `upper=` parameter.
* Two-panel, upper/lower: Define the `upper=` and `lower=` parameters.
* Two-panel, left/right: Define the `left=` and `upper=` parameters; the
  `upper=` section will be shown on the right.
* Three-panel: Define the `left=`, `upper=`, and `lower=` parameters; the
  `upper=` and `lower=` sections will be shown on the right.

To use a specific layout by default for a given LIT instance, pass the key
(e.g., "simple", "default", or the name of a custom layout) as a server flag
when initializing LIT (`--default_layout=<layout>`) or by setting the default
value for that flag in you `server.py` file, e.g.,
`flags.FLAGS.set_default('default_layout', 'my_layout_name')`. The layout can
also be set on-the-fly with the `layout=` URL param, which will take precedence.

Note: The pre-configured layouts are added to every `LitApp` instance using
[dictionary comprehension](https://docs.python.org/3/library/stdtypes.html#dict)
where the Mapping passed to the `LitApp` constructor overrides the
pre-configured layouts `Mapping`. Thus, you can remove or change these
pre-configured layouts as you like by passing a `Mapping` where the values of
`simple`, `default`, and/or `experimental` is `None` (to remove) or a
`LitCanonicalLayout` instance (to override) as you desire.

## Accessing the LIT UI in Notebooks

As an alternative to running a LIT server and connecting to it through a web
browser, LIT can be used directly inside of python notebook environments, such
as [Colab](https://colab.research.google.com/) and
[Jupyter](https://jupyter.org/).

After installing LIT through pip, create a `lit_nlp.notebook.LitWidget` object,
passing in a dict of models and a dict of datasets, similar to the
`lit_nlp.dev_server.Server` constructor. You can optionally provide a height
parameter that specifies the height in pixels to render the LIT UI.

Then, in its own output cell, call the `render` method on the widget object to
render the LIT UI. The LIT UI can be rendered in multiple cells if desired. The
LIT UI can also be rendered in its own browser tab, outside of the notebook, by
passing the parameter `open_in_new_tab=True` to the `render` method. The
`render` method can optionally take in a configuration object to specify
certain options to render the LIT UI using, such as the selected layout,
current display tab, dataset, and models. See
[notebook.py](https://github.com/PAIR-code/lit/blob/main/lit_nlp/notebook.py) for details.

The widget has a `stop` method which shuts down the widget's server. This can be
important for freeing up resources if you plan to create multiple LIT widget
instances in a single notebook. Stopping the server doesn't disable the model
and dataset instances used by the server; they can still be used in the notebook
and take up the resources they require.

Check out an
[example notebook](https://colab.research.google.com/github/pair-code/lit/blob/main/lit_nlp/examples/notebooks/LIT_sentiment_classifier.ipynb).

## Using LIT components outside of LIT

All LIT Python components (models, datasets, interpreters, metrics, generators,
etc.) are standalone classes that do not depend on the serving framework. You
can easily use them from Colab, in scripts, or in your libraries. This can
also be handy for development, as you can test new models or components without
needing to reload the server or click the UI.

For example, to view examples in a dataset:

```python
from lit_nlp.examples.datasets import glue
dataset = glue.SST2Data('validation')
print(dataset.examples)  # list of records {"sentence": ..., "label": ...}
```

And to run inference on a few of them:

```python
from lit_nlp.examples.models import glue_models

model = glue_models.SST2Model("/path/to/model/files")
preds = list(model.predict(dataset.examples[:5]))
# will return records {"probas": ..., "cls_emb": ..., ...} for each input
```

Or to compute input salience using
[LIME](https://homes.cs.washington.edu/~marcotcr/blog/lime/):

```python
from lit_nlp.components import lime_explainer

lime = lime_explainer.LIME()
lime.run([dataset.examples[0]], model, dataset)
# will return {"tokens": ..., "salience": ...} for each example given
```

For a full working example in Colab, see [LIT_components_example.ipynb](https://colab.research.google.com/github/pair-code/lit/blob/dev/lit_nlp/examples/notebooks/LIT_components_example.ipynb).


<!-- Links -->

[build-metadata]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/app.py
[components-py]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/dataset.py
[curves-interp]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/curves.py
[dataset-py]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/dataset.py
[grad-maps]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py
[json]: https://www.json.org
[mnli-dataset]: https://cims.nyu.edu/~sbowman/multinli/
[mnli-demo]: https://pair-code.github.io/lit/demos/glue.html
[model-py]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/dataset.py
[should_display_module]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/client/core/lit_module.ts
[types_py]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/api/types.py
[types_ts]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/client/lib/lit_types.ts
[utils-lib]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/client/lib/utils.ts
[word-replacer]: https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/word_replacer.py
