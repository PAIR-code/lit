---
title: LIT - Learn
layout: layouts/sub.liquid

hero-height: 245
hero-image: /assets/images/LIT_Tutorials_Banner.png
hero-title: "Model probing for understandable, reliable, and fair machine learning"
hero-copy: "Learn how to navigate LIT and how LIT enables analysis of NLP models. "

sub-nav: '<a href="#basics">Basics</a><a href="#analysis">Conducting analysis</a>'
color: "#fef0f7"
---

<div class="mdl-cell--8-col mdl-cell--8-col-tablet mdl-cell--4-col-phone">

<a name="basics"></a>

## Discover the basics

{% include partials/tutorial-link-element c-title: "A Tour of LIT", link: "/tutorials/tour",
c-copy: "Get familiar with the interface of the Language Interpretability Tool." %}

{% include partials/spacer height:50 %}

<a name="analysis"></a>

## Conducting analysis in LIT

{% include partials/tutorial-link-element c-title: "Exploring a Sentiment Classifier", link: "/tutorials/sentiment",
c-copy: "Learn about how we used LIT to analyze a sentiment classifier." %}

{% include partials/tutorial-link-element c-title: "Debugging a Text Generator", link: "/tutorials/generation",
c-copy: "Learn about how we used LIT to debug summarization by a text generation model." %}

{% include partials/tutorial-link-element c-title: "Gender Bias in Coreference", link: "/tutorials/coref",
c-copy: "Learn how we used LIT to explore gendered associations in a pronoun resolution model." %}

{% include partials/spacer height:50 %}

</div>