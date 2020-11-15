---
title: Exploring a Sentiment Classifier
layout: layouts/tutorial.liquid

hero-image: /assets/images/sample-banner.png
hero-title: "Sentiment Analysis"
hero-copy: "Learn about how we used LIT to analyze a sentiment classifier."

bc-anchor-category: "analysis"
bc-category-title: "Analysis"
bc-title: "Sentiment"

time: "3 minutes"
takeaways: "Learn about how the metrics table and saliency maps assisted an analysis of a sentiment classifier's performance when dealing with negation."
---

## Exploring a Sentiment Classifier

{% include partials/link-out link: "../../demos/glue.html", text: "Explore this demo yourself." %}

Or, run your own with [`examples/glue_demo.py`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/glue_demo.py)

How well does a sentiment classifier handle negation? We can use LIT to interactively ask this question and get answers. We loaded up LIT the development set of the Stanford Sentiment Treebank (SST), which contains sentences from movie reviews that have been human-labeled as having a negative sentiment (0), or a positive sentiment (1). For a model, we are using a BERT-based binary classifier that has been trained to classify sentiment.

Using the search function in LIT’s data table, we find the 67 datapoints containing the word “not”. By selecting these datapoints and looking at the Metrics Table, we find that our BERT model gets 91% of these correct, which is slightly higher than the accuracy across the entire dataset.


{% include partials/inset-image image: '/assets/images/lit-metrics-not.png',
  caption: 'Above: A comparison of metrics on datapoints containing "not" versus the entire dataset.'%}

But we might want to know if this is truly robust. We can select individual datapoints and look for explanations. For example, take the negative review, “It’s not the ultimate depression-era gangster movie.”. As shown below, salience maps suggest that “not” and “ultimate” are important to the prediction. We can verify this by creating modified inputs, using LIT’s datapoint editor. Removing “not” gets a strongly positive prediction from “It’s the ultimate depression-era gangster movie.”.

{% include partials/inset-image image: '/assets/images/lit-not-saliency.png',
  caption: 'Above: Prediction saliency of the original sentence, including "not".'%}

{% include partials/inset-image image: '/assets/images/lit-saliency.png',
  caption: 'Above: Prediction saliency of the altered sentence, with "not" removed.'%}

Using the LIT features of data table searching, the metrics table, salience maps, and manual editing, we’re able to show both in aggregate and in a specific instance, that our model handles negation correctly.

