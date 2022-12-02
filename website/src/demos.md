---
title: LIT - Demos
layout: layouts/sub.liquid

hero-image: /assets/images/LIT_Demos_Banner.png
hero-title: "Take LIT for a spin!"
hero-copy: "Get a feel for LIT in a variety of hosted demos."

color: "#49596c"
---

<div class="mdl-cell--8-col mdl-cell--8-col-tablet mdl-cell--4-col-phone">
  <div class="mdl-grid no-padding">
  {%  include partials/demo-card
      c-title: "Tabular data",
      link: "/demos/penguins.html",
      c-data-source: "Palmer Penguins",
      c-copy: "Analyze a tabular data model with LIT, including exploring partial dependence plots and automatically finding counterfactuals.",
      tags: "tabular, binary classification",
      external:"true" %}

  {%  include partials/demo-card
      c-title: "Image classification",
      link: "/demos/images.html",
      c-data-source: "Imagenette",
      c-copy: "Analyze an image classification model with LIT, including multiple image salience techniques.",
      tags: "images, multiclass classification",
      external:"true" %}

  {%  include partials/demo-card
      c-title: "Classification and regression models",
      link: "/demos/glue.html",
      c-data-source: "Stanford Sentiment Treebank,  Multi-Genre NLI Corpus, Semantic Textual Similarity Benchmark"
      c-copy: "Use LIT with any of three tasks from the General Language Understanding Evaluation (GLUE) benchmark suite. This demo contains binary classification (for sentiment analysis, using SST2), multi-class classification (for textual entailment, using MultiNLI), and regression (for measuring text similarity, using STS-B).",
      tags: "BERT, binary classification, multi-class classification, regression",
      external:"true" %}

  {%  include partials/external-demo-card
      c-title: "Notebook usage",
      link: "https://colab.research.google.com/github/PAIR-code/lit/blob/main/lit_nlp/examples/notebooks/LIT_sentiment_classifier.ipynb",
      c-data-source: "Stanford Sentiment Treebank"
      c-copy: "Use LIT directly inside a Colab notebook. Explore binary classification for sentiment analysis using SST2 from the General Language Understanding Evaluation (GLUE) benchmark suite.",
      tags: "BERT, binary classification, notebooks",
      external:"true" %}

  {%  include partials/demo-card
      c-title: "Gender bias in coreference systems",
      link: "/demos/coref.html",
      c-data-source: "Winogender schemas",
      c-copy: "Use LIT to explore gendered associations in a coreference system, which matches pronouns to their antecedents. This demo highlights how LIT can work with structured prediction models (edge classification), and its capability for disaggregated analysis.",
      tags: "BERT, coreference, fairness, Winogender",
      external:"true" %}

  {%  include partials/demo-card
      c-title: "Fill in the blanks",
      link: "/demos/lm.html",
      c-data-source: "Stanford Sentiment Treebank, Movie Reviews",
      c-copy: "Explore a BERT-based masked-language model. See what tokens the model predicts should fill in the blank when any token from an example sentence is masked out.",
      tags: "BERT, masked language model",
      external:"true" %}

  {%  include partials/demo-card
      c-title: "Text generation",
      link: "/demos/t5.html",
      c-data-source: "CNN / Daily Mail",
      c-copy: "Use a T5 model to summarize text. For any example of interest, quickly find similar examples from the training set, using an approximate nearest-neighbors index.",
      tags: "T5, generation",
      external:"true" %}

  {%  include partials/demo-card
      c-title: "Evaluating input salience methods",
      link: "/demos/is_eval.html",
      c-data-source: "Stanford Sentiment Treebank, Toxicity",
      c-copy: "Explore the faithfulness of input salience methods across different datasets and artificial shortcuts.",
      tags: "salience, evaluation",
      external:"true" %}
  </div>
</div>
