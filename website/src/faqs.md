---
title: LIT - FAQs
layout: layouts/sub.liquid

hero-image: /assets/images/LIT_FAQs_Banner.png
hero-title: "Frequently Asked Questions"

sub-nav: '<a href="https://github.com/pair-code/lit/issues/new" target="_blank">Ask a question</a>'
color: "#ebf6f7"
---

<div class="mdl-cell--8-col mdl-cell--8-col-tablet mdl-cell--4-col-phone">

{% include partials/faq-element 
  f-title: "Where do I go to report a bug in LIT?", 
  f-copy: "

Submit bugs, ask questions, suggest content, and request features on our [Github issues list](https://github.com/pair-code/lit/issues/new)." %}

{% include partials/faq-element 
  f-title: "How many datapoints can LIT handle?", 
  f-copy: "

The number of datapoints depends on the size of the individual datapoints themselves, along with the models being run and your hardware. 
The exact number depends on your model, and on your hardware. We've successfully tested with 10,000 examples (the full MultiNLI `validation_matched` split) including embeddings from our MNLI model.

Additionally, we're working on expanding the maximum number of datapoints that the tool can handle.
" %}

{% include partials/faq-element 
    f-title: "What kinds of models can LIT handle?", 
    f-copy: "
    
LIT can handle a variety of models, regardless of ML framework used, or output modality of the model.
It is not limited to a strict set of model types. For more information, see the [developers guide](https://github.com/PAIR-code/lit/blob/main/documentation/development.md)." %}

{% include partials/faq-element 
  f-title: "I have proprietary data. Is LIT secure for my team to use?", 
  f-copy: "
  
We don't store, collect or share datasets, models or any other information loaded into LIT. When you run a LIT server, anyone with access to the web address of the server will be able to see data from the loaded datasets and interact with the loaded models. If you need to restrict access to a LIT server, then make sure to configure the hosting of your LIT server to do so." %}

</div>
