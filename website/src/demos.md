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
  {%  include partials/demo-card,
      c-title: "Tabular data",
      link: "/demos/penguins.html",
      c-data-source: "Palmer Penguins",
      c-copy: "Analyze a tabular data model with LIT, including exploring partial dependence plots and automatically finding counterfactuals.",
      tags: "tabular, binary classification",
      external:"true" %}

  {%  include partials/demo-card,
      c-title: "Classification and regression models",
      link: "/demos/glue.html",
      c-data-source: "Stanford Sentiment Treebank,  Multi-Genre NLI Corpus, Semantic Textual Similarity Benchmark"
      c-copy: "Use LIT with any of three tasks from the General Language Understanding Evaluation (GLUE) benchmark suite. This demo contains binary classification (for sentiment analysis, using SST2), multi-class classification (for textual entailment, using MultiNLI), and regression (for measuring text similarity, using STS-B).",
      tags: "BERT, binary classification, multi-class classification, regression",
      external:"true" %}

  {%  include partials/external-demo-card,
      c-title: "Notebook usage",
      link: "https://colab.research.google.com/github/PAIR-code/lit/blob/main/lit_nlp/examples/notebooks/LIT_sentiment_classifier.ipynb",
      c-data-source: "Stanford Sentiment Treebank"
      c-copy: "Use LIT directly inside a Colab notebook. Explore binary classification for sentiment analysis using SST2 from the General Language Understanding Evaluation (GLUE) benchmark suite.",
      tags: "BERT, binary classification, notebooks",
      external:"true" %}
  </div>
</div>
