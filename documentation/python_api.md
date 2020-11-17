# Python APIs

<!--* freshness: { owner: 'lit-dev' } *-->

<!-- [TOC] placeholder - DO NOT REMOVE -->

This page describes the available APIs for LIT's Python backend. It assumes some
familarity with the basic [system design](development.md#design-overview) and
the [type system](development.md#type-system).

The following is intended to give a conceptual overview; for the most precise
documentation, see the code in [api](../lit_nlp/api)
and [examples](../lit_nlp/examples).

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
[type system](development.md#type-system)) to describe themselves to other
components.

For full examples, see
[examples](../lit_nlp/examples). In particular:

*   [`simple_tf2_demo.py`](../lit_nlp/examples/simple_tf2_demo.py)
    for a self-contained Keras/TF2 model for sentiment analysis.
*   [`simple_pytorch_demo.py`](../lit_nlp/examples/simple_pytorch_demo.py)
    for a self-contained PyTorch model for sentiment analysis.

## Datasets

Datasets ([`Dataset`](../lit_nlp/api/dataset.py)) are
just a list of examples, with associated type information following LIT's
[type system](development.md#type-system).

*   `spec()` should return a flat dict that describes the fields in each example
*   `self._examples` should be a list of flat dicts

Implementations should subclass
[`Dataset`](../lit_nlp/api/dataset.py). Usually this
is just a few lines of code - for example, the following is a complete dataset
loader for [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/):

```py
class MultiNLIData(Dataset):
  """Loader for MultiNLI development set."""

  NLI_LABELS = ['entailment', 'neutral', 'contradiction']

  def __init__(self, path):
    # Read the eval set from a .tsv file as distributed with the GLUE benchmark.
    df = pandas.read_csv(path, sep='\t')
    # Store as a list of dicts, conforming to self.spec()
    self._examples = [{
      'premise': row['sentence1'],
      'hypothesis': row['sentence2'],
      'label': row['gold_label'],
      'genre': row['genre'],
    } for _, row in df.iterrows()]

  def spec(self):
    return {
      'premise': lit_types.TextSegment(),
      'hypothesis': lit_types.TextSegment(),
      'label': lit_types.Label(vocab=self.NLI_LABELS),
      # We can include additional fields, which don't have to be used by the model.
      'genre': lit_types.Label(),
    }
```

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
*   `Dataset.remap(field_map: Dict[str, str])` will return a new `Dataset` with
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

Models ([`Model`](../lit_nlp/api/model.py)) are
functions which take inputs and produce outputs, with associated type
information following LIT's [type system](development.md#type-system). The core
API consists of three methods:

*   `input_spec()` should return a flat dict that describes necessary input
    fields
*   `output_spec()` should return a flat dict that describes the model's
    predictions and any additional outputs
*   `predict_minibatch()` and/or `predict()` should take a sequence of inputs
    (satisfying `input_spec()`) and yields a parallel sequence of outputs
    matching `output_spec()`.

Implementations should subclass
[`Model`](../lit_nlp/api/model.py). An example for
[MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) might look something like:

```py
class NLIModel(Model):
  """Wrapper for a Natural Language Inference model."""

  NLI_LABELS = ['entailment', 'neutral', 'contradiction']

  def __init__(self, model_path, **kw):
    # Load the model into memory so we're ready for interactive use.
    self._model = _load_my_model(model_path, **kw)

  ##
  # LIT API implementations
  def predict(self, inputs: List[Input]) -> Iterable[Preds]:
    """Predict on a single minibatch of examples."""
    examples = [self._model.convert_dict_input(d) for d in inputs]  # any custom preprocessing
    return self._model.predict_examples(examples)  # returns a dict for each input

  def input_spec(self):
    """Describe the inputs to the model."""
    return {
        'premise': lit_types.TextSegment(),
        'hypothesis': lit_types.TextSegment(),
    }

  def output_spec(self):
    """Describe the model outputs."""
    return {
      # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
      'probas': lit_types.MulticlassPreds(vocab=NLI_LABELS, parent='label'),
    }
```

Unlike the dataset example, this model implementation is incomplete - you'll
need to customize `predict()` (or `predict_minibatch()`) accordingly with any
pre- or post-processing needed, such as tokenization.

Note: The `Model` base class implements simple batching, aided by the
`max_minibatch_size()` function. This is purely for convenience, since most deep
learning models will want this behavior. But if you don't need it, you can
simply override the `predict()` function directly and handle large inputs
accordingly.

Note: there are a few additional methods in the model API - see
[`Model`](../lit_nlp/api/model.py) for details.

### Adding more outputs

The above example defined a black-box model, with predictions but no access to
internals. If we want a richer view into the model's behavior, we can add
additional return fields corresponding to hidden-state activations, gradients,
word embeddings, or attention. For example, a BERT-based model with several such
features might have the following `output_spec()`:

```py
  def output_spec(self):
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

LIT components and frontend modules will automatically detect these spec fields
and use them to support additional interpretation methods, such as the embedding
projector or gradient-based salience maps.

You can also implement multi-headed models this way: simply add additional
output fields for each prediction (such as another `MulticlassPreds`), and
they'll be automatically detected.

See the [type system documentation](development.md#type-system) for more details
on avaible types and their semantics.

## Interpretation Components

Backend interpretation components include metrics, salience maps, visualization
aids like [UMAP](https://umap-learn.readthedocs.io/en/latest/), and
counterfactual generator plug-ins.

Most such components implement the
[`Interpreter`](../lit_nlp/api/components.py) API.
Conceptually, this is any function that takes a set of datapoints and a model,
and produces some output.[^identity-component] For example,
[local gradient-based salience (GradientNorm)](../lit_nlp/components/gradient_maps.py)
processes the `TokenGradients` and `Tokens` returned by a model and produces a
list of scores for each token. The Integrated Gradients saliency method
additionally requires a `TokenEmbeddings` input and corresponding output, as
well as a label field `Target` to pin the gradient target to the same class as
an input and corresponding output. See the
[GLUE models class](../lit_nlp/examples/models/glue_models.py)
for an example of these spec requirements.

The core API involves implementing the `run()` method:

```python
  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    # config is any runtime options to this component, such as a threshold for
    # (binary) classification metrics.
```

Note: a more general `run_with_metadata()` method is also available; this
receives a list of `IndexedInput` which contain additional metadata, such as
parent pointers for tracking counterfactuals.

Output from an interpreter component is unconstrained; it's up to the frontend
component requesting it to process the output correctly. In particular, some
components (such as salience maps) may operate on each example independently,
similar to model predictions, while others (such as metrics) may produce
aggregate summaries of the input set.

Interpreters are also responsible for verifying compatibility by reading the
model and dataset specs; these are also used to determine what fields to operate
on. A typical implementation just loops over the relevant specs. For example,
for
[simple gradient-based salience](../lit_nlp/components/gradient_maps.py)
we might have:

```python
  def find_fields(self, output_spec: Spec) -> List[Text]:
    # Find TokenGradients fields
    grad_fields = utils.find_spec_keys(output_spec, types.TokenGradients)

    # Check that these are aligned to Tokens fields
    for f in grad_fields:
      tokens_field = output_spec[f].align  # pytype: disable=attribute-error
      assert tokens_field in output_spec
      assert isinstance(output_spec[tokens_field], types.Tokens)
    return grad_fields

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[List[JsonDict]]:
    """Run this component, given a model and input(s)."""
    # Find gradient fields to interpret
    output_spec = model.output_spec()
    grad_fields = self.find_fields(output_spec)
    logging.info('Found fields for gradient attribution: %s', str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      return None

    # do rest of the work to create the salience maps for each available field

    # return a dtypes.SalienceMap for each input, which has a list of
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
[`SimpleMetrics`](../lit_nlp/components/metrics.py)
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
              config: Optional[JsonDict] = None) -> Dict[Text, float]:
    """Compute metric(s) between labels and predictions."""
    del config
    mse = sklearn_metrics.mean_squared_error(labels, preds)
    pearsonr = scipy_stats.pearsonr(labels, preds)[0]
    spearmanr = scipy_stats.spearmanr(labels, preds)[0]
    return {'mse': mse, 'pearsonr': pearsonr, 'spearmanr': spearmanr}
```

The implementation of `SimpleMetrics.run()` uses the `parent` key (see
[type system](development.md#type-system)) in fields of the model's output spec
to find the appropriate input fields to compare against, and calls `compute()`
accordingly on the unpacked values.

### Generators

Conceptually, a generator is just an interpreter that returns new input
examples. These may depend on the input only, as for techniques such as
backtranslation, or can involve feedback from the model, such as for adversarial
attacks. Currently, generators use a separate API, subclassing
[`Generator`](../lit_nlp/api/components.py), but in
the near future this will be merged into the `Interpreter` API described above.

The core generator API is:

```python
class Generator(metaclass=abc.ABCMeta):
  """Base class for LIT generators."""

  def generate_all(self,
                   inputs: List[JsonDict],
                   model: lit_model.Model,
                   dataset: lit_dataset.Dataset,
                   config: Optional[JsonDict] = None) -> List[List[JsonDict]]:
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
API to support model-based generators (such as backtranslation) which benefit
from batched requests.

As with other interpreter components, a generator can take custom arguments
through `config`, such as the list of substitutions for the
[word replacer](../lit_nlp/components/word_replacer.py).

#### Backtranslator Generator

The [backtranslator](../lit_nlp/components/backtranslator.py)
generator translates text segment inputs into foreign languages and back to the
source language in order to create paraphrases.
It relies on the Google Cloud Translate API to perform those translations.
To use it, you must have a Google Cloud project and set up Cloud Translation
as described at https://cloud.google.com/translate/docs/setup.
Then, download  your application credentials file locally and set the
GOOGLE_APPLICATION_CREDENTIALS environment variable to point to that file.
With that environment variable set to the correct path, LIT can make use of the
backtranlator generator if you pass it as a generator in the Server constructor.

## LIT Application & Serving

TODO: write this
