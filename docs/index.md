# 🔥 Language Interpretability Tool

<!--* freshness: { owner: 'lit-dev' reviewed: '2020-08-04' } *-->



## Getting Started

Download the repo and set up a Python environment:

```sh
git clone https://github.com/PAIR-code/lit.git ~/lit
cd ~/lit
conda env create -f environment.yml
conda activate lit-nlp
```

Build the frontend (output will be in `~/lit/client/build`). You only need to do
this once, unless you change the TypeScript or CSS files.

```sh
cd ~/lit/client
yarn  # install deps
yarn build --watch
```

And run a LIT server, such as those included in
../lit_nlp/examples:

```sh
cd ~/lit
python -m lit_nlp.examples.pretrained_lm_demo --models=bert-base-uncased \
  --port=5432
```

You can then access the LIT UI at http://localhost:4321.

## User Guide

To learn about LIT's features, check out the [user guide](user_guide.md).

## Adding your own models or data

You can easily run LIT with your own model by creating a custom `demo.py`
launcher, similar to those in ../lit_nlp/examples. The basic steps
are:

*   Write a data loader which follows the
    [`lit.Dataset` API](python_api.md#datasets)
*   Write a model wrapper which follows the
    [`lit.Model` API](python_api.md#models)
*   Pass models, datasets, and any additional
    [components](python_api.md#interpretation-components) to the LIT server class

For a full walkthrough, see [adding models and data](python_api.md#adding-models-and-data).

## Extending LIT with new components

LIT is easy to extend with new interpretability components, generators, and
more, both on the frontend or the backend. See the
[developer guide](development.md) to get started.
