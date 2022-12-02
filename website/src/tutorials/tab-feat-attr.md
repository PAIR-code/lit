---
title: Tabular Feature Attribution
layout: layouts/tutorial.liquid

hero-image: /assets/images/sample-banner.png
hero-title: "Tabular Feature Attribution"
hero-copy: "Learn how to use the Tabular Feature Attribution module in LIT."

bc-anchor-category: "analysis"
bc-category-title: "Analysis"
bc-title: "Tabular Feature Attribution"

time: "15 minutes"
takeaways: "Learn how to use the Kernel SHAP based Tabular Feature Attribution module in LIT."
---

## Tabular Feature Attribution

{%  include partials/link-out
    link: "../../demos/penguins.html",
    text: "Explore this demo yourself." %}

Or, run your own with
[`examples/penguin_demo.py`](https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/penguin_demo.py)

LIT supports many techniques like salience maps and counterfactual generators
for text data. But what if you have a tabular dataset? You might want to find
out which features (columns) are most relevant to the model’s predictions. LIT's
Feature Attribution module for
[tabular datasets](https://github.com/PAIR-code/lit/wiki/components.md#tabular-data)
support identification of these important features. This tutorial provides a
walkthrough for this module within LIT, on the
[Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/).

{%  include partials/info-box
    title: 'Kernel SHAP based Feature Attribution',
    text: "The Feature Attribution functionality is
        [achieved using SHAP](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf).
        In particular LIT uses
        [Kernel SHAP](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)
        over tabular data, which is basically a specially weighted local linear
        regression for estimating SHAP values and works for any model. For now,
        the feature attribution module is only shown in the UI when working with
        [tabular data](https://github.com/PAIR-code/lit/wiki/components.md#tabular-data)."%}

### **Overview**

The [penguins demo](https://pair-code.github.io/lit/demos/penguins.html) is a
simple classifier for predicting penguin species from the Palmer Penguins
dataset. It classifies the penguins as either Adelie, Chinstrap, or Gentoo based
on 6 features&mdash;body mass (g), [culmen](https://en.wikipedia.org/wiki/Beak#Culmen)
depth (mm), culmen length (mm), flipper length (mm), island, and sex.

{% include partials/info-box title: 'Filtering out incomplete data points',
  text: "Palmer Penguins is a tabular dataset with 344 penguin specimens. LIT’s
  penguin demo filters out 11 of these penguins due to missing information (sex
  is missing for all penguins, though some are missing additional information),
  resulting in 333 data points being loaded for analysis."%}

The Feature Attribution module shows up in the bottom right of the demo within
the Explanations tab. It computes
[Shapley Additive exPlanation (SHAP)](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)
values for each feature in a set of inputs and displays these values in a table.
The controls for this module are:

1.  The **sample size slider,** which defaults to a value of 30. SHAP
    computations are very expensive and it is infeasible to compute them for the
    entire dataset. Through testing, we found that 30 is about the maximum
    number of samples we can run SHAP on before performance takes a significant
    hit, and it becomes difficult to use above 50 examples. Clicking the Apply
    button will automatically check the Show attributions from the Tabular SHAP
    checkbox, and LIT will start computing the SHAP values.
2.  The **prediction key** selects the model output value for which influence is
    computed. Since the penguin mode only predicts one feature, species, this is
    set to species and cannot be changed. If a model can predict multiple values
    in different fields, for example predicting species and island or species
    and sex, then you could change which output field to explain before clicking
    Apply.
3.  The **heatmap toggle** can be enabled to color code the SHAP values.
4.  The **facets button** and **show attributions for selection checkbox**
    enable conditionally running the Kernel SHAP interpreter over subsets of the
    data. We will get into the specifics of this with an example later on in
    this tutorial.

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-1.png',
    caption: 'An overview of the Penguins demo, notice the tabular feature
        attribution (1) and salience maps (2) modules in the bottom right and
        center, respectively.'%}

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-2.png',
    caption: 'The tabular feature attribution module has three main elements of
        interactivity: an expansion panel where you can configure the SHAP
        parameters (1), a heatmap toggle to activate color the cells in the
        results table based on the scores (2), and a facets control for
        exploring subsets of the data (3).'%}

#### **A Simple Use Case : Feature Attribution for 10 samples**

To get started with the module, we set sample size to a small value, 10, and
start the SHAP computation with heatmap enabled.

{% include partials/info-box title: 'Edge cases for the sample size button',
  text: "Kernel SHAP computes feature importance relative to a pseudo-random
  sample of the dataset. The sample size is set with the slider, and the samples
  are drawn from either the current selection (i.e., a subset of the data that
  were manually selected or are included as part of a slice) or the entire
  dataset. When sampling from the current selection, the sample size can have
  interesting edge cases:

* If the selection is empty, LIT samples the “sample size” number of data points
  from the entire dataset.
* If the sample size is zero or larger than the selection, then LIT computes
  SHAP for the entire selection and does not sample additional data from the
  dataset.
* If sample size is smaller than the selection, then LIT samples the “sample
  size” number of data points from the selected inputs."%}

Enabling the heatmap provides a visual indicator of the polarity and strength of
a feature's influence. A reddish hue indicates negative attribution for that
particular feature and a bluish hue indicates positive attribution. The deeper
the color the stronger its influence on the predictions.

{%  include partials/info-box
    title: 'Interpreting salience polarity',
    text: "Salience is always relative to the model's prediction of one class.
        Intuitively, a positive attribution score for a feature of an example
        means that if this feature was removed we expect a drop in model
        confidence in the prediction of this class. Similarly, removing a
        feature with a negative score would correspond to an increase in the
        model's confidence in the prediction of this class."%}

SHAP values are computed per feature per example, from which LIT computes the
mean, min, median, and max feature values across the examples. The min and max
values can be used to spot any outliers during analysis. The difference between
the mean and the median can be used to gain more insights about the
distribution. All of this enables statistical comparisons and will be enhanced
in future releases of LIT.

Each of the columns in the table can be sorted using the up (ascending) or down
(descending) arrow symbols in the column headers. The table is sorted in
ascending alphabetical order of input feature names (field) by default. If there
are many features in a dataset this space will get crowded, so LIT offers a
filter button for each of the columns to look up a particular feature or value
directly.

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-4.png',
    caption: 'Start by reducing the sample size from 30 to 10, this will speed
        up the SHAP computations.'%}

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-5.png',
    caption: 'The results of the SHAP run over a sample of 10 inputs from the
        entire dataset. Notice how subtle the salience values are in the "mean"
        column.'%}

#### **Faceting & Binning of Features**

Simply speaking, facets are subsets of the dataset based on specific feature
values. We can use facets to explore differences in SHAP values between subsets.
For example, instead of looking at SHAP values from 10 samples containing both
male and female penguins, we can look at male penguins and female penguins
separately by faceting based on sex. LIT also allows you to select multiple
features for faceting, and it will generate the facets by feature crosses. For
example, if you select both sex (either male or female) and island (one of
Biscoe, Dream and Torgersen), then LIT will create 6 facets for (Male, Biscoe),
(Male, Dream), (Male, Torgersen), (Female, Biscoe), (Female, Dream), (Female,
Torgersen) and show the SHAP values for whichever facets have a non-zero number
of samples.

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-6.png',
    caption: 'Each facet of the dataset is given its own expansion panel. Click
        on the down arrow on the right to expand the section and see the results
        for that facet.'%}

Numerical features support more complex faceting options. Faceting based on
numerical features allows for defining bins using 4 methods: discrete, equal
intervals, quantile, and threshold. Equal intervals will evenly divide the
feature’s [domain](https://en.wikipedia.org/wiki/Domain_of_a_function) into N
equal-sized bins. Quantile will create N bins that each contain (approximately)
the same number of examples. Threshold creates two bins, one for the examples
with values up to and including the threshold value, and one for examples with
values above the threshold value. The discrete method requires specific dataset
or model spec configuration, and we do not recommend using that method with this
demo.

Categorical and boolean features do not have controllable binning behavior. A
bin is created for each label in their vocabulary.

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-7.png',
    caption: 'Clicking the facets button will open the configuration controls.
        Use these to configure how divide the dataset into subsets.'%}

LIT supports as many as 100 facets (aka bins). An indicator in the faceting
config dialog lets you know how many would be created given the current
settings.

Faceting is not supported for selections, meaning that if you already have a
selection of elements (let’s say 10 penguins), then facets won’t split it
further.

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-9.png',
    caption: 'LIT limits the number of facets to 100 bins for performance
        reasons. Attempting to exceed this limit will cause the active features
        to highlight red so you can adjust their configurations.'%}

### **Side-by-side comparison : Salience Maps Vs Tabular Feature Attribution**

The Feature Attribution module works well in conjunction with other modules. In
particular, we are going to look at the Salience Maps module which allows us to
enhance our analysis. Salience Maps work on one data point at a time, whereas
the Tabular Feature Attribution usually looks at a set of data points.

{% include partials/info-box title: 'Slightly different color scales',
  text: "The color scales are slightly different between the salience maps
  module and the tabular feature attribution module. Salience maps use a
  gamma-adjusted color scale to make values more prominent."%}

#### **One random data point**

In this example, a random data point is chosen using the select random button in
the top right corner and the unselected data points are hidden in the Data
Table. After running both the salience maps module and the feature attribution
module for the selected point, we can see that the values in the mean column of
Tabular SHAP output match the saliency scores exactly. Note also that the mean,
min, median and max values are all the same when a single datapoint is selected.

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-11.png',
    caption: 'The results in the tabular feature attribution and salience maps
        modules will be the same for single datapoint selections.'%}

#### **A slice of 5 random data points**

LIT uses a [complex selection model](https://github.com/PAIR-code/lit/blob/main/documentation/ui_guide.md#datapoint-selections)
and different modules react to it differently. Salience Maps only care about the
primary selection (the data point highlighted in a deep cyan hue in the data
table) in a slice of elements, whereas Feature Attribution uses the entire list
of selected elements.

{%  include partials/info-box
    title: 'Using Salience Maps to support Tabular Feature Attribution',
    text: "Changing primary selection reruns SHAP in the Salience Maps module
        but not in Tabular Feature Attribution. So, we can effectively toggle
        through the items in our selection one-by-one and see how they compare
        to the mean values in the Feature Attribution module. Another thing to
        note is that the Salience Maps module supports comparison between a
        pinned datapoint and the primary selection, so we can do the above
        comparisons in a pair-wise manner as well."%}

As we can see in this example, where we run both modules on a slice of 5
elements, the Salience Maps module is only providing its output for the primary
selection (data point 0), whereas the Tabular Feature Attribution module is
providing values for the entire selection by enabling the “Show attributions for
selection” checkbox. This allows us to use the salience map module as a kind of
magnifying glass to focus on any individual example even when we are considering
a slice of examples in our exploration of the dataset.

{%  include partials/inset-image
    image: '/assets/images/tab-feat-attr-image-12.png',
    caption: 'The salience maps module is a great way to compare the scores for
        each datapoint in a selection against the scores for that entire
        selection from. the tabular feature attribution module.'%}

### **Conclusion**

Tabular Feature Attribution based on Kernel SHAP allows LIT users to explore
their tabular data and find the most influential features affecting model
predictions. It also integrates nicely with the Salience Maps module to allow
for fine-grained inspections. This is the first of many features in LIT for
exploring tabular data, and more exciting updates would be coming in future
releases!
