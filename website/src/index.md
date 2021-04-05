---
title: Language Interpretability Tool
layout: layouts/main.liquid
---

<div class="mdl-cell--8-col mdl-cell--12-col-tablet mdl-cell--8-col-phone">

{% include partials/display1 text:"The Language Interpretability Tool (LIT) is an open-source platform for <strong>visualization and understanding of NLP models</strong>." %}

{% include partials/home-cta-button text:"Take a tour", link:"/tutorials/tour" %}
{% include partials/home-cta-button text:"Setup LIT", link:"/setup" %}

{% include partials/spacer height:60 %}

</div>

![overview of LIT]({% root %}/assets/images/lit-tweet.gif)

{% include partials/spacer height:60 %}

<div class="mdl-cell--8-col mdl-cell--12-col-tablet mdl-cell--8-col-phone">

The Language Interpretability Tool (LIT) is for researchers and practitioners looking to understand NLP model behavior through a visual, interactive, and extensible tool.  

Use LIT to ask and answer questions like:
- What kind of examples does my model perform poorly on?
- Why did my model make this prediction? Can it attribute it to adversarial behavior, or undesirable priors from the training set?
- Does my model behave consistently if I change things like textual style, verb tense, or pronoun gender?

LIT contains many built-in capabilities but is also customizable, with the ability to add custom interpretability techniques, metrics calculations, counterfactual generators, visualizations, and more.

For a similar tool to explore general-purpose machine learning models, check out the [What-If Tool](https://whatif-tool.dev).

LIT can be run as a standalone server, or inside of python notebook environments such as Colab and Jupyter.
</div>

{% include partials/spacer height:50 %}

{% include partials/display2 text:"Flexible and powerful model probing" %}

<div class="mdl-grid no-padding">

{% include partials/one-of-three-column title:"Built-in capabilities", text: "

Salience maps

Attention visualization

Metrics calculations

Counterfactual generation

Model and datapoint comparison

Embedding visualization

And more...

" %}
{% include partials/one-of-three-column title:"Supported task types", text: "

Classification

Regression

Text generation / seq2seq

Masked language models

Span labeling

Multi-headed models

And more...

" %}
{% include partials/one-of-three-column title:"Framework agnostic", text: "

TensorFlow 1.x

TensorFlow 2.x

PyTorch

Notebook compatibility

Custom inference code

Remote Procedure Calls

And more...

" %}

</div>

{% include partials/spacer height:50 %}

## What’s the latest

<div class="mdl-grid no-padding">
  {% include partials/home-card image: '/assets/images/LIT_Contribute.png', action: 'CODE', 
  title: 'Contribute to LIT', desc: 'LIT is open to anyone who wants to help develop and improve it!', 
  cta-text:"View developer guide", link: 'https://github.com/PAIR-code/lit/blob/main/documentation/development.md', external:"true" %}
  
  {% include partials/home-card image: '/assets/images/LIT_Updates.png', action: 'UPDATES', 
  title: 'Latest updates', desc: 'New features, updates, and improvements to LIT.', 
  cta-text:"See release notes", link: 'https://github.com/PAIR-code/lit/blob/main/RELEASE.md' external:"true" %}

  {% include partials/home-card image: '/assets/images/LIT_Paper.png', action: 'RESEARCH', 
  title: 'Demo Paper at EMNLP ‘20', desc: 'Read about what went into LIT in our demo paper, presented at EMNLP ‘20.', 
  cta-text:"Read the paper", link: 'https://www.aclweb.org/anthology/2020.emnlp-demos.15.pdf' external:"true" %}

</div>
