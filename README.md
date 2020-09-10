# 🔥 Language Interpretability Tool (LIT)

<!--* freshness: { owner: 'lit-dev' reviewed: '2020-08-04' } *-->

The Language Interpretability Tool (LIT) is a visual, interactive
model-understanding tool for NLP models.

LIT is built to answer questions such as:

*   **What kind of examples** does my model perform poorly on?
*   **Why did my model make this prediction?** Can this prediction be attributed
    to adversarial behavior, or to undesirable priors in the training set?
*   **Does my model behave consistently** if I change things like textual style,
    verb tense, or pronoun gender?

![Example of LIT UI](docs/images/figure-1.png)

LIT supports a variety of debugging workflows through a browser-based UI.
Features include:

*   **Local explanations** via salience maps, attention, and rich visualization
    of model predictions.
*   **Aggregate analysis** including custom metrics, slicing and binning, and
    visualization of embedding spaces.
*   **Counterfactual generation** via manual edits or generator plug-ins to
    dynamically create and evaluate new examples.
*   **Side-by-side mode** to compare two or more models, or one model on a pair
    of examples.
*   **Highly extensible** to new model types, including classification,
    regression, span labeling, seq2seq, and language modeling. Supports
    multi-head models and multiple input features out of the box.
*   **Framework-agnostic** and compatible with TensorFlow, PyTorch, and more.

For a broader overview, check out [our paper](https://arxiv.org/abs/2008.05122) and the
[user guide](docs/user_guide.md).

## Documentation

*   [User Guide](docs/user_guide.md)
*   [Developer Guide](docs/development.md)
*   [FAQ](docs/faq.md)

## Download and Installation

LIT can be installed via pip, or can be built from source. Building from source
is necessary if you wish to update any of the front-end or core back-end code.

### Install from source

Download the repo and set up a Python environment:

```sh
git clone https://github.com/PAIR-code/lit.git ~/lit

# Set up Python environment
cd ~/lit
conda env create -f environment.yml
conda activate lit-nlp
conda install cudnn cupti  # optional, for GPU support
conda install -c pytorch pytorch  # optional, for PyTorch

# Build the frontend
cd ~/lit/lit_nlp/client
yarn && yarn build
```

Note: if you see [an error](https://github.com/yarnpkg/yarn/issues/2821)
running yarn on Ubuntu/Debian, be sure you have the
[correct version installed](https://yarnpkg.com/en/docs/install#linux-tab).

### pip installation

```sh
pip install lit-nlp
```

The pip installation will install all necessary prerequisite packages for use
of the core LIT package. It also installs the code to run our demo examples.
It does not install the prerequisites for those demos, so you need to install
those yourself if you wish to run the demos. See
[environment.yml](./environment.yml) for the list of all packages needed for
running the demos.

## Running LIT

### Quick-start: sentiment classifier

```sh
cd ~/lit
python -m lit_nlp.examples.quickstart_sst_demo --port=5432
```

This will fine-tune a [BERT-tiny](https://arxiv.org/abs/1908.08962) model on the
[Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html),
which should take less than 5 minutes on a GPU. After training completes, it'll
start a LIT server on the development set; navigate to http://localhost:5432 for
the UI.

### Quick start: language modeling

To explore predictions from a pretrained language model (BERT or GPT-2), run:

```sh
cd ~/lit
python -m lit_nlp.examples.pretrained_lm_demo --models=bert-base-uncased \
  --port=5432
```

And navigate to http://localhost:5432 for the UI.

### More Examples

See [lit_nlp/examples](./lit_nlp/examples). Run similarly to the above:

```sh
cd ~/lit
python -m lit_nlp.examples.<example_name> --port=5432 [optional --args]
```

## User Guide

To learn about LIT's features, check out the [user guide](docs/user_guide.md), or
watch this [short video](https://www.youtube.com/watch?v=j0OfBWFUqIE).

## Adding your own models or data

You can easily run LIT with your own model by creating a custom `demo.py`
launcher, similar to those in [lit_nlp/examples](./lit_nlp/examples). The basic
steps are:

*   Write a data loader which follows the
    [`Dataset` API](docs/python_api.md#datasets)
*   Write a model wrapper which follows the [`Model` API](docs/python_api.md#models)
*   Pass models, datasets, and any additional
    [components](docs/python_api.md#interpretation-components) to the LIT server
    class

For a full walkthrough, see
[adding models and data](docs/python_api.md#adding-models-and-data).

## Extending LIT with new components

LIT is easy to extend with new interpretability components, generators, and
more, both on the frontend or the backend. See the
[developer guide](docs/development.md) to get started.

## Citing LIT

If you use LIT as part of your work, please cite [our paper](https://arxiv.org/abs/2008.05122):

```
@misc{tenney2020language,
    title={The Language Interpretability Tool: Extensible, Interactive Visualizations and Analysis for NLP Models},
    author={Ian Tenney and James Wexler and Jasmijn Bastings and Tolga Bolukbasi and Andy Coenen and Sebastian Gehrmann and Ellen Jiang and Mahima Pushkarna and Carey Radebaugh and Emily Reif and Ann Yuan},
    year={2020},
    eprint={2008.05122},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Disclaimer

This is not an official Google product.

LIT is a research project, and under active development by a small team.
There will be some bugs and rough edges, but we're releasing v0.1 because we
think it's pretty useful already. We want LIT to be an open platform, not a
walled garden, and we'd love your suggestions and feedback - drop us a line in
the [issues](https://github.com/pair-code/lit/issues).
