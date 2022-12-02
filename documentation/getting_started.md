# Getting Started with LIT

<!--* freshness: { owner: 'lit-dev' reviewed: '2022-08-17' } *-->

<!-- [TOC] placeholder - DO NOT REMOVE -->

## Hosted demos

If you want to jump in and start playing with the LIT UI, check out
https://pair-code.github.io/lit/demos/ for links to our hosted demos.

For a guide to the many features available, check out the
[UI guide](./ui_guide.md) or this
[short video](https://www.youtube.com/watch?v=j0OfBWFUqIE).

## LIT with your model <!-- DO NOT REMOVE {#custom-demos} -->

LIT provides a simple [Python API](./api.md) for use with custom models and
data, as well as components such as metrics and counterfactual generators. Most
LIT users will take this route, which involves writing a short `demo.py` binary
to link in `Model` and `Dataset` implementations and configure the server. In
most cases this can be just a few lines:

```python
  datasets = {
      'foo_data': FooDataset('/path/to/foo.tsv'),
      'bar_data': BarDataset('/path/to/bar.tfrecord'),
  }
  models = {'my_model': MyModel('/path/to/model/files')}
  lit_demo = lit_nlp.dev_server.Server(models, datasets, port=4321)
  lit_demo.serve()
```

Check out the [API documentation](./api.md#adding-models-and-data) for more, and
the [demos directory](./demos.md) for a wealth of examples. The
[components guide](./components.md) also gives an overview of interpretability
methods and other features available in LIT, and describes how to enable each
for your task.

## Using LIT in notebooks <!-- DO NOT REMOVE {#colab} -->

LIT can also be used directly from Colab and Jupyter notebooks, with the LIT UI
rendered in an output cell. See https://colab.research.google.com/github/pair-code/lit/blob/dev/lit_nlp/examples/notebooks/LIT_sentiment_classifier.ipynb for an example.

Note: if you see a 403 error in the output cell where LIT should render, you may
need to enable cookies on the Colab site, or pass a custom `port=` to the
`LitWidget` constructor.

## Stand-alone components <!-- DO NOT REMOVE {#standalone} -->

Many LIT components - such as models, datasets, metrics, and salience methods -
are stand-alone Python classes and can be easily used outside of the LIT UI. For
additional details, see the
[API documentation](./api.md#using-components-outside-lit) and an example Colab
at https://colab.research.google.com/github/pair-code/lit/blob/dev/lit_nlp/examples/notebooks/LIT_components_example.ipynb.

## Run an existing example <!-- DO NOT REMOVE {#running-lit} -->

The [demos page](./demos.md) lists some of the pre-built demos available for a
variety of model types. The code for these is under [lit_nlp/examples](../lit_nlp/examples)
;
each is a small script that loads one or more models and starts a LIT server.

Most demos can be run with a single blaze command. To run the default one, you
can do:

```sh
python -m lit_nlp.examples.glue_demo \
  --quickstart --port=4321 --alsologtostderr
```

Then navigate to https://localhost:4321 to access the UI.

For most models we recommend using a GPU, though the `--quickstart` flag above
loads a set of smaller models that run well on CPU. You can also pass
`--warm_start=1.0`, and LIT will run inference and cache the results before
server start.

For an overview of supported model types and frameworks, see the
[components guide](./components.md).
