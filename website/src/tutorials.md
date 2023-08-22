---
title: LIT - Tutorials
layout: layouts/sub.liquid

hero-height: 245
hero-image: /assets/images/LIT_Tutorials_Banner.png
hero-title: "Model probing for understandable, reliable, and fair NLP"
hero-copy: "Learn how to navigate LIT and use it to analyze different types of models. "

color: "#fef0f7"
---

<div class="mdl-cell--8-col mdl-cell--8-col-tablet mdl-cell--4-col-phone">

<a name="basics"></a>

## Discover the basics

{% include partials/tutorial-link-element c-title: "A Tour of LIT", link: "/tutorials/tour",
c-copy: "Get familiar with the interface of the Learning Interpretability Tool." %}

{% include partials/spacer height:50 %}

<a name="analysis"></a>

## Conducting analysis in LIT

{% include partials/tutorial-link-element c-title: "Salience Maps for Text", link: "/tutorials/text-salience",
c-copy: "Learn how to use salience maps for text data in LIT." %}

{% include partials/tutorial-link-element c-title: "Tabular Feature Attribution", link: "/tutorials/tab-feat-attr",
c-copy: "Learn how to use the Kernel SHAP based Tabular Feature Attribution module in LIT." %}

{% include partials/tutorial-link-element c-title: "Global Model Analysis with TCAV", link: "/tutorials/tcav",
c-copy: "Learn about examining model behavior through user-curated concepts." %}

{% include partials/tutorial-link-element c-title: "Exploring a Sentiment Classifier", link: "/tutorials/sentiment",
c-copy: "Learn about how we used LIT to analyze a sentiment classifier." %}

{% include partials/tutorial-link-element c-title: "Debugging a Text Generator", link: "/tutorials/generation",
c-copy: "Learn about how we used LIT to debug summarization by a text generation model." %}

{% include partials/tutorial-link-element c-title: "Gender Bias in Coreference", link: "/tutorials/coref",
c-copy: "Learn how we used LIT to explore gendered associations in a pronoun resolution model." %}

{% include partials/spacer height:50 %}

</div>
