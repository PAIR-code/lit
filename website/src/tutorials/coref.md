---
title: Gender Bias in Coreference
layout: layouts/tutorial.liquid

hero-image: /assets/images/sample-banner.png
hero-title: "Gender Bias in Coreference"
hero-copy: "Learn how we can use LIT to explore gendered associations in a pronoun resolution model."

bc-anchor-category: "analysis"
bc-category-title: "Analysis"
bc-title: "Coreference"

time: "10 minutes"
takeaways: "Learn about how to explore fairness using datapoint comparison and metrics comparisons."
---

## Gender Bias in Coreference

{% include partials/link-out link: "../../demos/coref.html", text: "Explore this demo yourself." %}

Or, run your own with [`examples/coref/coref_demo.py`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/coref/coref_demo.py)

Does a system encode gendered associations, which might lead to incorrect predictions? We explore this for coreference, the task of identifying whether two mentions refer to the same (real-world) entity. For example, in the sentence "The technician told the customer that they could pay with cash.", we understand from the context that "they" refers to "the customer", the one paying.

The Winogender dataset introduced by Rudinger et al. 2018 presents a set of challenging coreference examples designed to explore gender bias. It consists of 120 templates, each with semantic context that makes it easy for humans to tell the answer. Each template is instantiated with different pronouns, in order to give a minimal pair:
- "The technician told the customer that **he** could pay with cash."
- "The technician told the customer that **she** could pay with cash."

In both cases, the pronoun should refer to the customer - but does our model agree? Or does it fall back on stereotypes about who can be a technician, or a customer? We can use LIT to explore this interactively, making use of the side-by-side functionality, structured prediction visualization, and powerful features for aggregate analysis to validate our findings.

We load our coreference model into LIT, along with a copy of the Winogender dataset. Our model predicts probabilities for each mention pair - in this case the (occupation, pronoun) and (participant, pronoun) pairs - and LIT renders this as a pair of edges:

{% include partials/inset-image image: '/assets/images/lit-coref-pred.png', 
  caption: 'Above: A coreference prediction.'%}

We can select an example by clicking the row in the data table in the top left of the UI; the predictions will display automatically in the "Predictions" tab below. To look at two predictions side-by-side, we can enable "Compare datapoints" mode in the toolbar, which will pin our first selection as a "reference" and allow us to select another point to compare:


{% include partials/inset-image image: '/assets/images/lit-coref-select.png', 
  caption: 'Above: Selecting two datapoints to compare.'%}

We see that LIT automatically replicates the predictions view, allowing us to see how our model handles "he" and "she" differently on these two sentences:

{% include partials/inset-image image: '/assets/images/lit-coref-compare.png', 
  caption: 'Above: Comparing coreference predictions of two datapoints.'%}

To see why this might be, we can make use of some additional information from the U.S. Bureau of Labor Statistics (BLS), which tabulates the gender percentages in different occupations. Our example loads this along with the dataset, and LIT shows this as a column in the data table:

{% include partials/inset-image image: '/assets/images/lit-coref-data.png', 
  caption: 'Above: Datapoints with extra informational feature columns.'%}

We see that "technician" is only 40% female, suggesting that our model might be picking up on social biases with its eagerness to identify "he" as the technician in the example above.

Is this a pattern? In addition to individual instances, we can use LIT to see if this holds on larger slices of the data. Turning to the "Performance" tab, we see that our model gets around 63% accuracy overall.

{% include partials/inset-image image: '/assets/images/lit-coref-metric-top.png', 
  caption: 'Above: Overall model accuracy in the metrics table.'%}

Let's see how this breaks down. On the right, the Scalars module lets us select data based on scalar values, such as the percent female of each profession according to BLS. Let's select the points on the left, with professions that are stereotypically male (< 25% female). Additionally, we'll stratify our metrics based on the pronoun group, and whether the answer should be the occupation term or the other, neutral, participant:

{% include partials/inset-image image: '/assets/images/lit-coref-metrics.png', 
  caption: 'Above: Metrics faceted into sub-groups and scalar results plots.'%}

We can see that on this slice, our model performs very well when the ground truth agrees with the stereotype - i.e. when the answer is the occupation term, our model resolves male pronouns correctly 91% of the time, while only matching female pronouns 37% of the time in exactly the same contexts.
