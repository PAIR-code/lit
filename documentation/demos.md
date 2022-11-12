# Demos

<!--* freshness: { owner: 'lit-dev' reviewed: '2022-09-08' }
      css: "//depot/google3/third_party/py/lit_nlp/g3doc/includes/demos.css"
*-->

<!-- [TOC] placeholder - DO NOT REMOVE -->

The LIT team maintains a number of hosted demos, as well as pre-built launchers
for some common tasks and model types.

For publicly-visible demos hosted on Google Cloud, see
https://pair-code.github.io/lit/demos/.

--------------------------------------------------------------------------------

## Classification <!-- DO NOT REMOVE {#classification .demo-section-header} -->

### Sentiment and NLI <!-- DO NOT REMOVE {#glue .demo-header} -->

**Hosted instance:** https://pair-code.github.io/lit/demos/glue.html \
**Code:** [lit_nlp/examples/glue_demo.py](../lit_nlp/examples/glue_demo.py)

*   Multi-task demo:
    *   Sentiment analysis as a binary classification task
        ([SST-2](https://nlp.stanford.edu/sentiment/treebank.html)) on single
        sentences.
    *   Natural Language Inference (NLI) using
        [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/), as a three-way
        classification task with two-segment input (premise, hypothesis).
    *   STS-B textual similarity task (see [Regression / Scoring](#scoring)
        below).
    *   Switch tasks using the Settings (⚙️) menu.
*   BERT models of different sizes, built on HuggingFace TF2 (Keras).
*   Supports the widest range of LIT interpretability features:
    *   Model output probabilities, custom thresholds, and multiclass metrics.
    *   Jitter plot of output scores, to find confident examples or ones near
        the margin.
    *   Embedding projector to find clusters in representation space.
    *   Integrated Gradients, LIME, and other salience methods.
    *   Attention visualization.
    *   Counterfactual generators, including HotFlip for targeted adversarial
        perturbations.

Tip: check out a case study for this demo on the public LIT website:
https://pair-code.github.io/lit/tutorials/sentiment

### Multilingual (XNLI) <!-- DO NOT REMOVE {#xnli .demo-header} -->

**Code:** [lit_nlp/examples/xnli_demo.py](../lit_nlp/examples/xnli_demo.py)

*   [XNLI](https://cims.nyu.edu/~sbowman/xnli/) dataset translates a subset of
    MultiNLI into 14 different languages.
*   Specify `--languages=en,jp,hi,...` flag to select which languages to load.
*   NLI as a three-way classification task with two-segment input (premise,
    hypothesis).
*   Fine-tuned multilingual BERT model.
*   Salience methods work with non-whitespace-delimited text, by using the
    model's wordpiece tokenization.

--------------------------------------------------------------------------------

## Regression / Scoring <!-- DO NOT REMOVE {#scoring .demo-section-header} -->

### Textual Similarity (STS-B) <!-- DO NOT REMOVE {#stsb .demo-header} -->

**Hosted instance:** https://pair-code.github.io/lit/demos/glue.html?models=stsb&dataset=stsb_dev \
**Code:** [lit_nlp/examples/glue_demo.py](../lit_nlp/examples/glue_demo.py)

*   STS-B textual similarity task, predicting scores on a range from 0
    (unrelated) to 5 (very similar).
*   BERT models built on HuggingFace TF2 (Keras).
*   Supports a wide range of LIT interpretability features:
    *   Model output scores and metrics.
    *   Scatter plot of scores and error, and jitter plot of true labels for
        quick filtering.
    *   Embedding projector to find clusters in representation space.
    *   Integrated Gradients, LIME, and other salience methods.
    *   Attention visualization.

--------------------------------------------------------------------------------

## Sequence-to-Sequence <!-- DO NOT REMOVE {#seq2seq .demo-section-header} -->

### Open-source T5 <!-- DO NOT REMOVE {#t5 .demo-header} -->

**Code:** [lit_nlp/examples/t5_demo.py](../lit_nlp/examples/t5_demo.py)

*   Supports HuggingFace TF2 (Keras) models as well as TensorFlow SavedModel
    formats.
*   Visualize beam candidates and highlight diffs against references.
*   Visualize per-token decoder hypotheses to see where the model veers away
    from desired output.
*   Filter examples by ROUGE score against reference.
*   Embeddings from last layer of model, visualized with UMAP or PCA.
*   Task wrappers to handle pre- and post-processing for summarization and
    machine translation tasks.
*   Pre-loaded eval sets for CNNDM and WMT.

Tip: check out a case study for this demo on the public LIT website:
https://pair-code.github.io/lit/tutorials/generation

--------------------------------------------------------------------------------

## Language Modeling <!-- DO NOT REMOVE {#lm .demo-section-header} -->

### BERT and GPT-2 <!-- DO NOT REMOVE {#bert .demo-header} -->

**Code:** [lit_nlp/examples/lm_demo.py](../lit_nlp/examples/lm_demo.py)

*   Compare multiple BERT and GPT-2 models side-by-side on a variety of
    plain-text corpora.
*   LM visualization supports different modes:
    *   BERT masked language model: click-to-mask, and query model at that
        position.
    *   GPT-2 shows left-to-right hypotheses for each target token.
*   Embedding projector to show latent space of the model.

--------------------------------------------------------------------------------

## Structured Prediction <!-- DO NOT REMOVE {#structured .demo-section-header} -->

### Gender Bias in Coreference <!-- DO NOT REMOVE {#coref .demo-header} -->

**Code:** [lit_nlp/examples/coref/coref_demo.py](../lit_nlp/examples/coref/coref_demo.py)

*   Gold-mention coreference model, trained on
    [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19).
*   Evaluate on the Winogender schemas
    ([Rudinger et al. 2018](https://arxiv.org/abs/1804.09301)) which test for
    gendered associations with profession names.
*   Visualizations of coreference edges, as well as binary classification
    between two candidate referents.
*   Stratified metrics for quantifying model bias as a function of pronoun
    gender or Bureau of Labor Statistics profession data.

Tip: check out a case study for this demo on the public LIT website:
https://pair-code.github.io/lit/tutorials/coref

--------------------------------------------------------------------------------

## Multimodal <!-- DO NOT REMOVE {#multimodal .demo-section-header} -->

### Tabular Data: Penguin Classification <!-- DO NOT REMOVE {#penguin .demo-header} -->

**Code:** [lit_nlp/examples/penguin_demo.py](../lit_nlp/examples/penguin_demo.py)

*   Binary classification on
    [penguin dataset](https://www.tensorflow.org/datasets/catalog/penguins).
*   Showing using of LIT on non-text data (numeric and categorical features).
*   Use partial-dependence plots to understand feature importance on individual
    examples, selections, or the entire evaluation dataset.
*   Use binary classifier threshold setters to find best thresholds for slices
    of examples to achieve specific fairness constraints, such as demographic
    parity.

### Image Classification with MobileNet <!-- DO NOT REMOVE {#mobilenet .demo-header} -->

**Code:** [lit_nlp/examples/image_demo.py](../lit_nlp/examples/image_demo.py)

*   Classification on ImageNet labels using a MobileNet model.
*   Showing using of LIT on image data.
*   Explore results of multiple gradient-based image saliency techniques in the
    Salience Maps module.

