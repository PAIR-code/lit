# Learning Interpretability Tool Release Notes

## Release 1.1.1

This release covers various improvements for sequence salience, including new
features in the UI module, support of more LLMs, and detailed tutorial and
documentation on how to use the sequence salience module for prompt engineering.

### New stuff
* New features in the sequence salience UI module -
[62f18b2](https://github.com/PAIR-code/lit/commit/62f18b2ff62bf77fa47205cffddf0d072a73c366),
[f0417c9](https://github.com/PAIR-code/lit/commit/f0417c93da282a4699253f335c2643be5e50567f),
[fe5a705](https://github.com/PAIR-code/lit/commit/fe5a705bfb013ac782e87351af08bc5b03204e71),
[1ec8626](https://github.com/PAIR-code/lit/commit/1ec8626da0e2a1922fb7812913f2677b232043ef),
[15184a1](https://github.com/PAIR-code/lit/commit/15184a18da69dacfb657c238ef8f5bac79ed7863),
[84af141](https://github.com/PAIR-code/lit/commit/84af141c7cf8a6ddb4db6ececec787ac235ddd17),
[27cafd8](https://github.com/PAIR-code/lit/commit/27cafd85636b3d18f40d15a01ffd5d0857ff0daa),
[3591e61](https://github.com/PAIR-code/lit/commit/3591e614fb09264ee03ae0c73510f1d0a4b74cdf),
[d108b59](https://github.com/PAIR-code/lit/commit/d108b596658f456f43e0b19473ab1c70c59cc065),
[309c4f2](https://github.com/PAIR-code/lit/commit/309c4f283af559ca34570e044d77d5c4a7cce540),
[99821d3](https://github.com/PAIR-code/lit/commit/99821d3b5505d857f919fe2455830e6c2338fd68),
[c8ee224](https://github.com/PAIR-code/lit/commit/c8ee224a445f925a9a7d6d7dc4472436190d0174)

* Support of more models (GPT2, Gemma, Llama, Mistral) on deep learning frameworks (Tensorflow, Pytorch) for Keras and Hugging Face -
[b26256a](https://github.com/PAIR-code/lit/commit/b26256a7c339c9e0940eb7a806528da23098ed03),
[45887d3](https://github.com/PAIR-code/lit/commit/45887d35d3880289595613224524157b19481ac0),
[b9941ed](https://github.com/PAIR-code/lit/commit/b9941ed7aea315022426710ccd32e8e1c7ff6c04),
[5ee7064](https://github.com/PAIR-code/lit/commit/5ee7064ec23933c41b4233061f9cc65b851fa7bb),
[8ea325b](https://github.com/PAIR-code/lit/commit/8ea325b292b09aecbb074abc877525dcdf4f4cd0)

* A tutorial to use sequence salience at [our website](https://pair-code.github.io/lit/tutorials/) and documentation updates -
[962faaa](https://github.com/PAIR-code/lit/commit/962faaabcf209f9cf024df5cc9684d8d4e4e64d8),
[96eff29](https://github.com/PAIR-code/lit/commit/96eff29198e69a8a9f2203d88f9027f5596f1614),
[f731e6d](https://github.com/PAIR-code/lit/commit/f731e6dfdeeb26f959022ed5aeda71e4f1f377d0),
[f4d7cac](https://github.com/PAIR-code/lit/commit/f4d7cacda3399d3e474420facd1a75d5b6af4824),
[49e7736](https://github.com/PAIR-code/lit/commit/49e77369fabd820b3f213f2d846efb5e81dbeafe)

### Non-breaking Changes, Bug Fixes, and Enhancements
* Py typing fix -
[d70e3d3](https://github.com/PAIR-code/lit/commit/d70e3d3c64671dfd5da034d3a47a34aedeef6469)
* Improvements on the curves UI module -
[3d61a09](https://github.com/PAIR-code/lit/commit/3d61a09b684b3d57fc23b1362091d5293d8e6d19)
[2efe62b](https://github.com/PAIR-code/lit/commit/2efe62b77dbfddf698b0a79408c0227bd21dc959)
* Support model-column search in LIT Data Table -
[525bf5e](https://github.com/PAIR-code/lit/commit/525bf5e7c005fe1f931867bc1206b527544865b3)
* Obsolete code cleanup -
[82abec6](https://github.com/PAIR-code/lit/commit/82abec688836b8e6d136de83e56660ea055dc91d)


## Release 1.1

This release provides the capabilities to interpret and debug the behaviors of
Generative AI models in LIT. Specifically, we added sequence salience, which
explains the impact of the preceding tokens on the generated tokens produced by
the GenAI models. Major changes include:
* An `LM salience` module in the LIT UI that computes generations, tokenization,
and sequence salience on-demand;
* Computation of sequence salience at different granularities, from the smallest
possible level of tokens, to more interpretable larger spans, such as words,
sentences, lines, or paragraphs.
* Support of OSS modeling frameworks, including KerasNLP and Hugging Face
Transformers for sequence salience computation.
This release would not have been possible without the work of our contributors.
Many thanks to:
[Ryan Mullins](https://github.com/RyanMullins),
[Ian Tenney](https://github.com/iftenney),
[Bin Du](https://github.com/bdu91), and
[Cibi Arjun](https://github.com/cpka145).

### New Stuff
* LM salience module in the LIT UI -
[ab294bd](https://github.com/PAIR-code/lit/commit/ab294bd3e15675c0e63e5a16ffe4b8cd4941c94f)
[5cffc4d](https://github.com/PAIR-code/lit/commit/5cffc4d933e611587b00c25861c911d5f734fa22)
[40bb57a](https://github.com/PAIR-code/lit/commit/40bb57a2531257c38137188090a24e70d47581c8)
[d3980cc](https://github.com/PAIR-code/lit/commit/d3980cc5414e1f9be895defc4f967bee8a2480fc)
[406fbc7](https://github.com/PAIR-code/lit/commit/406fbc7690ee72f6f96ecf68f1238822ae8951c2)
[77583e7](https://github.com/PAIR-code/lit/commit/77583e74236aa443a21ad0779b0ab9c023821b93)
[a758f98](https://github.com/PAIR-code/lit/commit/a758f98c5153f23955b0190a75dc1258ba57b645)
* Sequence salience for decoder-only LM, with support for GPT-2 and KerasNLP -
[27e6901](https://github.com/PAIR-code/lit/commit/27e6901164044c0d33658603369a55600da0b202)
[80cf699](https://github.com/PAIR-code/lit/commit/80cf699f92cd77d58cb2a2a60b9314010b1f336c)
[1df3ba8](https://github.com/PAIR-code/lit/commit/1df3ba8449e865edb5806c10c8054c246d1e38e3)
[b6ab352](https://github.com/PAIR-code/lit/commit/b6ab3522b301810cab3c75723f3fe0dabf829577)
[c97a710](https://github.com/PAIR-code/lit/commit/c97a710416538906ea6b269f90264c0602a15593)
* Prompt examples for sequence salience -
[4f19891](https://github.com/PAIR-code/lit/commit/4f1989180ee570642285682f843242be5bffb9ef)
[000c844](https://github.com/PAIR-code/lit/commit/000c84486ed61439c98dbfdd92959bdbb6f5119f)
[34aa110](https://github.com/PAIR-code/lit/commit/34aa110c36fe0c7ec670f06662078d2f572c79c6)
[ca032ff](https://github.com/PAIR-code/lit/commit/ca032ffb3196e71fd0a7a09118635ca6dafc8153)


### Non-breaking Changes, Bug Fixes, and Enhancements
* Improvements to display various fields and their default ranges -
[8a3f366](https://github.com/PAIR-code/lit/commit/8a3f366816833ead164ecfca778b465ef6d074bb)
[e63b674](https://github.com/PAIR-code/lit/commit/e63b67484fc7f4dbfa3484126c355350d2127bf7)
[d274508](https://github.com/PAIR-code/lit/commit/d2745088966c4ac31a3755f55096eeb8193c5a91)
* Allow only displaying the UI layouts provided by users -
[a219863](https://github.com/PAIR-code/lit/commit/a21986342d83ae64d58607e337fab9db7736242a)
* Internal dependency changes -
[f254fa8](https://github.com/PAIR-code/lit/commit/f254fa8500d6267278fa3dc32fb4bbf56beb7cf7)
[724bdee](https://github.com/PAIR-code/lit/commit/724bdee1f9ea45ce998b9031eea4ad1169299efb)
[2138bd9](https://github.com/PAIR-code/lit/commit/2138bd920e72553f9c920ba489962c8649738574)
* Fix issues with adding more than one example from counterfactual generators -
[d4302bd](https://github.com/PAIR-code/lit/commit/d4302bd6bfc7e4c778ba0e96397ac620242a8d21)
* Fix issues with loading `SimpleSentimentModel` -
[ac8ed59](https://github.com/PAIR-code/lit/commit/ac8ed5902a2c96019ea1137b5138d48017fabf4e)
* Notebook widget improvements -
[cdf79eb](https://github.com/PAIR-code/lit/commit/cdf79eb9048be3e6798e916d5e1ac4cc294929b0)
* Docs updates

## Release 1.0

This is a major release, covering many new features and API changes from the
`dev` branch since the v0.5 release over 8 months ago. This release includes
a variety of breaking changes meant to simplify various aspects of the LIT API
and visual changes to improve usability. This release includes over 250 commits.
Major changes include:

* Refactored python code to remove `_with_metadata` methods from all component
  and model classes.
* Refactored Model and BatchedModel python classes to remove `predict_minibatch`
  method.
* Reworked UI and backend logic for dynamic loading of new datasets and models
  from the UI. This makes use of the new `init_spec` methods for datasets and
  model classes.
  * Added a blank demo with no models or datasets preloaded which allows for
    dynamic loading of models and datasets through the UI.
* Refactored to upgrade metrics calculation from a type of interpreter to its
  own top-level concept.
* Updated front-end layout code to default to a new layout that includes a
  full height side-panel on the left side to complement the existing top and
  bottom panels, providing for more customization of module layouts.
* Added automatic metrics calculations for multilabel models.
* Added target selector dropdown for saliency methods.
* A visual redesign of the Salience Clustering module.
* Improved searching capabilities in the Data Table module.
* Improved the Data Table module's display of long strings through a "Show more"
  capability.
* Updated to Python 3.10.
* Updated to Node 18 and Typescript 5.0.
* Improved documentation pages, now at https://pair-code.github.io/lit/documentation/


This release would not have been possible without the work of our new
contributors in 2023. Many thanks to
[Minsuk Kahng](https://github.com/minsukkahng),
[Nada Hussein](https://github.com/nadah09),
[Oscar Wahltinez](https://github.com/owahltinez),
[Bin Du](https://github.com/bdu91), and
[Cibi Arjun](https://github.com/cpka145)
for your support and contributions to this project!
A full list of contributors to this repo can be found at https://github.com/PAIR-code/lit/graphs/contributors.

### Breaking Changes
* Adds init_spec() capability to models and datasets for dynamic loading -
  [d28eec3](https://github.com/PAIR-code/lit/commit/d28eec3b00737282e353230c99a25cd656897958),
  [d624562](https://github.com/PAIR-code/lit/commit/d624562931b001902cbeda474b21ece9208fad66),
  [7bb60b2](https://github.com/PAIR-code/lit/commit/7bb60b24aa0ec7755d9e763f8689a3558409e5bc),
  [f74798a](https://github.com/PAIR-code/lit/commit/f74798aedf2d10c9fcda5309e302ff3581b73a95),
  [db51d9d](https://github.com/PAIR-code/lit/commit/db51d9d8e20705978337d13ef288e38e28f44b62),
  [f3b0d6e](https://github.com/PAIR-code/lit/commit/f3b0d6eb9746397a8b535379adedb8bfd728dead),
  [0f133cf](https://github.com/PAIR-code/lit/commit/0f133cfdf92a86c81761d82369d560b464b69790),
  [9eebe57](https://github.com/PAIR-code/lit/commit/9eebe5748d8973a778b7c93c80d7d522c2945185),
  [bcc6c09](https://github.com/PAIR-code/lit/commit/bcc6c090a29f832288b336a832ac0912ed9116e8),
  [99e78ff](https://github.com/PAIR-code/lit/commit/99e78ffd87758600652c6819f5a9416c642ea9fd)
* Simplify Model spec code -
  [16b72f7](https://github.com/PAIR-code/lit/commit/16b72f7b9923c09b195e0a5f62132b1baeb9cce1)
* Promote Metrics to top-level property of LitMetadata -
  [f019279](https://github.com/PAIR-code/lit/commit/f0192796ffc6630336aa5e4dfef99aeacfaa90c7),
  [6ba1db8](https://github.com/PAIR-code/lit/commit/6ba1db8e4c7fcfe59e7a42c6b9e02887d21bf658),
  [c1777ea](https://github.com/PAIR-code/lit/commit/c1777eadf34ac615c9f00c4cdd6ca13e8f91794b)
* Remove _with_metadata and batched methods from models and components -
  [cb4f6b0](https://github.com/PAIR-code/lit/commit/cb4f6b01717768e1050a2a8b8a7356265ad3e9fe),
  [e020faa](https://github.com/PAIR-code/lit/commit/e020faa9ec9a5c8755814e9a7fc707b640e73492),
  [e9ce692](https://github.com/PAIR-code/lit/commit/e9ce692980fc048b351a3fb31f59c9d3c3e3c5bf),
  [5f1a971](https://github.com/PAIR-code/lit/commit/5f1a97149a03e8fdea5f0082ad7446c99c40f756),
  [061973a](https://github.com/PAIR-code/lit/commit/061973aa2c197d7b13d787da33b2d0d33ed9dcda),
  [ad65fd9](https://github.com/PAIR-code/lit/commit/ad65fd9584735f51ec9bbfa6e32dcf69024d43b0),
  [bc6f82b](https://github.com/PAIR-code/lit/commit/bc6f82b2d8477140e8ede54be233fd012c6d53f0),
  [7888c66](https://github.com/PAIR-code/lit/commit/7888c6677081049111f1c3d51943dea2c9351c59),
  [9767670](https://github.com/PAIR-code/lit/commit/976767089fbbb56ab14056c298bd0e5480b20486),
  [0ec1527](https://github.com/PAIR-code/lit/commit/0ec152786c5858822222901c2bea09ce3e5af036),
  [e30e59a](https://github.com/PAIR-code/lit/commit/e30e59a6d560bd5102b61a6b90e8251b33931228),
  [b29d1f3](https://github.com/PAIR-code/lit/commit/b29d1f393b165a1f9b39bcbfe1e13caa36c075cd),
  [5ed93bd](https://github.com/PAIR-code/lit/commit/5ed93bd59e9b540d173a6f3048a2c4f6f993f642),
  [5047bdd](https://github.com/PAIR-code/lit/commit/5047bddd617bff6314986133590f6bd5b6845faf),
  [a15cc88](https://github.com/PAIR-code/lit/commit/a15cc88222325cd539c204c1df7be395d7a07814),
  [0146d5f](https://github.com/PAIR-code/lit/commit/0146d5f101391cf31df0756bca1494107f0e50f6),
  [50fc3a4](https://github.com/PAIR-code/lit/commit/50fc3a4397d7f3ba2f004886990c77b7f5523747),
  [6fdcbfe](https://github.com/PAIR-code/lit/commit/6fdcbfe09e521439991d6e487c6b7ef61c69a170),
  [ce38565](https://github.com/PAIR-code/lit/commit/ce38565ecd0370361c78f332c3ac8813cd416b63)
* Simplifications and refactors in layout system -
  [4a5c0cb](https://github.com/PAIR-code/lit/commit/4a5c0cb8836bce4f483df82e0d09b75868c487be),
  [5f6a46a](https://github.com/PAIR-code/lit/commit/5f6a46a8a15fe557b8ec16b5e2898817100b3bc4),
  [2551c2c](https://github.com/PAIR-code/lit/commit/2551c2ced40a58e17b7b2bc2fd4bc06530090cb8),
  [cc7bfd5](https://github.com/PAIR-code/lit/commit/cc7bfd54456c4a3ef4a3d9c832e2bf06b3c63947),
  [fb2467d](https://github.com/PAIR-code/lit/commit/fb2467d9152cc2a9e5ee113ff0f6796db9a71808)
* Update LIT to Node 18 and TypeScript 5.0 -
  [7b96a6d](https://github.com/PAIR-code/lit/commit/7b96a6d7b42785a184752381f4d684f8923bff9d)
* Update LIT to Python 3.10 -
  [8bce86a](https://github.com/PAIR-code/lit/commit/8bce86a27dfc27dad37d8d2ebcab96cb8cdfde5e)



### New Stuff
* Add three-panel layout configuration option -
  [a95ed67](https://github.com/PAIR-code/lit/commit/a95ed67100f24163624edb4bb659ccfa871dc9bf)
* Add output embeddings and attention options to GlueConfig -
  [6e0df41](https://github.com/PAIR-code/lit/commit/6e0df41636405b4ee5556cbf797fcce5887c6070)
* Allow downloading/copying data from the slice editor - 
  [57fac3a](https://github.com/PAIR-code/lit/commit/57fac3aeb98fa49c508b20837eded3f4ec80e8f9)
* Use new custom tooltip elemement in various places -
  [d409900](https://github.com/PAIR-code/lit/commit/d409900984336d4f8ac73735b1fff57c92623ca4),
  [bd0f7fc](https://github.com/PAIR-code/lit/commit/bd0f7fc47682b16dd4c8e530e17b1a295def1433),
  [6c25619](https://github.com/PAIR-code/lit/commit/6c2561994db506586b63de46a5900dd5dc6c0078),
  [7d30408](https://github.com/PAIR-code/lit/commit/7d3040819cfda82fe5ac2ce5b9fc46556918da20),
  [6779a4b](https://github.com/PAIR-code/lit/commit/6779a4b1fcba64bc0d8174ea46e58e6a7684af53),
  [9179c73](https://github.com/PAIR-code/lit/commit/9179c730b73a7defc746fdd775c5a0ce78d40e84)
* Add multi-label metrics to LIT -
  [c0e3663](https://github.com/PAIR-code/lit/commit/c0e3663156991ae3639e1ee707d613705f60f6f8)
* Improved UI for dynamic loading of models and datasets -
  [abc8d1a](https://github.com/PAIR-code/lit/commit/abc8d1a37ae14626211467f72a129f35415a1887),
  [b7ce560](https://github.com/PAIR-code/lit/commit/b7ce56037c27880b6d1c2ed27dce449c6a8d26ad)
* Replace conda installation instructions with pip
  [de23ceb](https://github.com/PAIR-code/lit/commit/de23ceb7c6801c71c63c253da669aab694c6c2c3)
* Add a new Blank demo for dynamic loading of models and datasets -
  [22b0dea](https://github.com/PAIR-code/lit/commit/22b0dea22a8167db7965ef8aec0b5c7e8b7509da)
* Add target-selector dropdowns to salience map module -
  [4c9a7ec](https://github.com/PAIR-code/lit/commit/4c9a7ecfc1a3d7a9fd3247b125d2e9c0d30a11f0),
  [10926ea](https://github.com/PAIR-code/lit/commit/10926ea2759db7881264b0b21924899cfb39de23),
  [f635ea7](https://github.com/PAIR-code/lit/commit/f635ea7a8548c8934db583a8a8f45bd63d38bd0a),
  [fe121ca](https://github.com/PAIR-code/lit/commit/fe121cabd240aff0bd08a9ba4a030dbd7ce12193),
  [8cb965a](https://github.com/PAIR-code/lit/commit/8cb965a78616f9ec7de133871ecf01d92a71293e)


### Non-breaking Changes, Bug Fixes, and Enhancements
* Fixes Scalars Module resize bug affecting datasets with scalar fields -
  [453461a](https://github.com/PAIR-code/lit/commit/453461a06b73b982b2db778ce05db8199d89193a)
* Moves Model-Dataset compatibility checks to Model class instead of ModelSpec -
  [c268ce4](https://github.com/PAIR-code/lit/commit/c268ce4890a627bc7c85d9fc277785b2d9d8ed85)
* Updates to the Salience Clustering module -
  [3a3aad3](https://github.com/PAIR-code/lit/commit/3a3aad302fcb97a89f43645bc81e2dd8fdeb3bfd),
  [7d3f235](https://github.com/PAIR-code/lit/commit/7d3f235fc9aad221f4c24b34545511f81eab9223),
  [ff759ad](https://github.com/PAIR-code/lit/commit/ff759ad313852844479d3f395a6c291ed18d3dce),
  [20ec052](https://github.com/PAIR-code/lit/commit/20ec052af16cc46f9a6159e2ada3ce6d03eda6f0)
* Data Table module improvements -
  [c7fa619](https://github.com/PAIR-code/lit/commit/c7fa619d921af92e34195d17b969596101dd24e0),
  [7301b28](https://github.com/PAIR-code/lit/commit/7301b28bd4456e0b9a981c7fd1e0dbc405d2b318),
  [dd23083](https://github.com/PAIR-code/lit/commit/dd23083f945cb660af9214d304be3b5045c5231d),
  [42d189a](https://github.com/PAIR-code/lit/commit/42d189a49f9e1c7afc3a92eda4e365994ff454fc),
  [1cc6964](https://github.com/PAIR-code/lit/commit/1cc696465a24fb5aaeb8f35e25b62fd673488555),
  [ab7da61](https://github.com/PAIR-code/lit/commit/ab7da61ef7f74441dad2b62b6065d5b1ff6f4d4c),
  [ee54333](https://github.com/PAIR-code/lit/commit/ee543339ca89aeb88648f68dfb2b09c87ecea145),
  [d74f2d6](https://github.com/PAIR-code/lit/commit/d74f2d626c62d0b1f8a76416bf6e3cb65cdb9429),
  [35487fa](https://github.com/PAIR-code/lit/commit/35487fa93d1987fc9a7eb98e2d20e3372e24f469),
  [ea25e75](https://github.com/PAIR-code/lit/commit/ea25e75a65f143b5a8c0ca9e4e71003d9a88b46e),
  [8c4bf1f](https://github.com/PAIR-code/lit/commit/8c4bf1ff998867540ae14f551bff2b5df64effd7),
  [ddf8e52](https://github.com/PAIR-code/lit/commit/ddf8e522a55e1ee60042ff2c54bb234f5a87106f)
* Various styling fixes, bug fixes, and code cleanup efforts
* Docs, FAQ, and README updates

## Release 0.5

This is a major release, covering many new features from the `dev` branch since
the v0.4 release nearly 11 months ago. Most notably, we're renaming! It's still
LIT, but now the L stands for "Learning" instead of "Language", to better
reflect the scope of LIT and support for non-text modalities like images and
tabular data. Additionally, we've made lots of improvements, including:

* New modules including salience clustering, tabular feature attribution, and
  a new Dive module for data exploration (inspired by our prior work on
  [Facets Dive](https://pair-code.github.io/facets/)).
* New demos and tutorials for input salience comparison and tabular feature
  attribution.
* Many UI improvements, with better consistency across modules and shared
  functionality for colors, slicing, and faceting of data.
* Better performance on large datasets (up to 100k examples), as well as
  improvements to the type system and new validation routines (`--validate`) for
  models and datasets.
* Download data as CSV directly from tables in the UI, and in notebook mode
  access selected examples directly from Python.
* Update to Python 3.9 and TypeScript 4.7.

This release would not have been possible without the work of many new
contributors in 2022. Many thanks to
[Crystal Qian](https://github.com/cjqian),
[Shane Wong](https://github.com/jswong65),
[Anjishnu Mukherjee](https://github.com/iamshnoo),
[Aryan Chaurasia](https://github.com/aryan1107),
[Animesh Okhade](https://github.com/animeshokhade),
[Daniel Levenson](https://github.com/dleve123),
[Danila Sinopalnikov](https://github.com/sinopalnikov),
[Deepak Ramachandran](https://github.com/DeepakRamachandran),
[Rebecca Chen](https://github.com/rchen152),
[Sebastian Ebert](https://github.com/eberts-google), and
[Yilei Yang](https://github.com/yilei)
for your support and contributions to this project!

### Breaking Changes

* Upgraded to Python 3.9 –
  [17bfabd](https://github.com/PAIR-code/lit/commit/17bfabd75959feae4d64e79db695fe38be7a14b0)
* Upgraded to Typescript 4.7 –
  [10e2548](https://github.com/PAIR-code/lit/commit/10e25480d43ecfa1800ed77fd5e2b49b69723c39)
* Layout definitions moved to Python –
  [05824c8](https://github.com/PAIR-code/lit/commit/05824c88296e9fed48ed6757b2f459ff6cc29968),
  [d3d19d2](https://github.com/PAIR-code/lit/commit/d3d19d2fbada9c12ab06630494c7cc84f9b3a9c8),
  [2994d7e](https://github.com/PAIR-code/lit/commit/2994d7e00582cff528e3753b43ce81ced00a1b30),
  [b78c962](https://github.com/PAIR-code/lit/commit/b78c96227bc760bb5009a1ed119b8fd568076767),
  [0eacdd0](https://github.com/PAIR-code/lit/commit/0eacdd026d2a0933f67d8aa2b5a1ec9d37a0d2d6)
* Moving classification and regression results to Interpreters –
  [2b4e622](https://github.com/PAIR-code/lit/commit/2b4e622922ba35df79e538c3a157356b854a54c6),
  [bcdbb80](https://github.com/PAIR-code/lit/commit/bcdbb8050ed1cdcd6350a556bbc394e67d4113fe),
  [dad8edb](https://github.com/PAIR-code/lit/commit/dad8edb8f05af8e4c3e46c352ee689988ec5cc11)
* Use a Pinning construct instead of comparison mode –
  [05bfc90](https://github.com/PAIR-code/lit/commit/05bfc906c91b3b748ffc7f3b414a046629ca16b1),
  [d7bdc65](https://github.com/PAIR-code/lit/commit/d7bdc654f147f879dec97e96f25d95d963fb7caa),
  [6a4ca00](https://github.com/PAIR-code/lit/commit/6a4ca0018211ed52e7eb24ec3d01ca4c683f179a),
  [0fe3c79](https://github.com/PAIR-code/lit/commit/0fe3c79352832c594a82a8e52d853c1c29742910),
  [5b2b737](https://github.com/PAIR-code/lit/commit/5b2b73767a2fb81f90c222786ed2a73b9171969d)
* Parallel, class-based Specs and LitTypes in Python and TypeScript code
    * Prep work –
      [db1ef3d](https://github.com/PAIR-code/lit/commit/db1ef3ddc7bd35df8c75325b9662fa41facf4359),
      [c85e556](https://github.com/PAIR-code/lit/commit/c85e556eedddf555449cd8e92b3218503b46dbb4),
      [660b8ef](https://github.com/PAIR-code/lit/commit/660b8ef3d47430e71fc0f9fcfad32a0e7b360557),
      [db58fa4](https://github.com/PAIR-code/lit/commit/db58fa42d18e605d997dce84f0b08797cc2729dc),
      [c020d25](https://github.com/PAIR-code/lit/commit/c020d2535a10ea137e25ea5ba87fa6d3d4cecc58),
      [eb02465](https://github.com/PAIR-code/lit/commit/eb024651e3b09e8bcd836e3558b6cef7e7b70160),
      [72edd26](https://github.com/PAIR-code/lit/commit/72edd26ed4f71d6b8d81ecefa5d09b508a29861d),
      [65c5b8a](https://github.com/PAIR-code/lit/commit/65c5b8a93643d4735c51e6ded48dcb3434203e60),
      [abb8889](https://github.com/PAIR-code/lit/commit/abb88890898848bb5a8fbe84f184a4b2b3a244cf),
      [4c93b62](https://github.com/PAIR-code/lit/commit/4c93b62da400ae30a86b65e415bd495f3e611449),
      [40d14e5](https://github.com/PAIR-code/lit/commit/40d14e5985c8dcde384de0b9f5bc469239e269f0),
      [9ec5324](https://github.com/PAIR-code/lit/commit/9ec53248e8c7b0a2e1ba0996e6084709ce2080ea),
      [40a661e](https://github.com/PAIR-code/lit/commit/40a661edafc71e1a0ae4f2d88eeb529d04c1172a)
    * Breaking changes to front-end typing infrastructure –
      [8c6ac11](https://github.com/PAIR-code/lit/commit/8c6ac1174cd1020c00491736a3d0fa78e05e0eed),
      [2522e4f](https://github.com/PAIR-code/lit/commit/2522e4f72e96c09a019630623b9061e73b4dce54),
      [0f8ff8e](https://github.com/PAIR-code/lit/commit/0f8ff8e251aee27654a9e1590c50aa5f75598edc),
      [58970de](https://github.com/PAIR-code/lit/commit/58970de691dea2be533e9c80e52768b2eb7b8f07),
      [ef72bfc](https://github.com/PAIR-code/lit/commit/ef72bfc4fcfc2bde06db0db0a7f105e9401d4cd2),
      [ccbb72c](https://github.com/PAIR-code/lit/commit/ccbb72c60d1eefc71c1eeca50408613cb65e445c),
      [a5b9f65](https://github.com/PAIR-code/lit/commit/a5b9f658188339c11c28fd43dbe25ff167e06c0b),
      [ab1e06a](https://github.com/PAIR-code/lit/commit/ab1e06a016fd7b309ee77237adf35f52d43e52d6),
      [853edd0](https://github.com/PAIR-code/lit/commit/853edd0b03f695aaa5d708312325dc13758070da),
      [cb528f1](https://github.com/PAIR-code/lit/commit/cb528f1bd502edf9f6ed25734a1ef81cfbff007b),
      [a36a936](https://github.com/PAIR-code/lit/commit/a36a936689443b4ed2417299e17dcd5a0b49de39),
      [74b5dbb](https://github.com/PAIR-code/lit/commit/74b5dbbb23259df7c3233cfcedce588ef62def82),
      [e811359](https://github.com/PAIR-code/lit/commit/e811359cabd092bacf14799ab811c314f6a8bf84)
    * Build fixes –
      [948adb3](https://github.com/PAIR-code/lit/commit/948adb3d35894cbd78cc73ddbe2ea8da5a883ace)
* Minimizing duplication in modules
    * Classification Results –
      [4f2b53d](https://github.com/PAIR-code/lit/commit/4f2b53d94c73e210a1def9043623590e077ee1b8)
    * Scalars, including its migration to Megaplot –
      [353b96e](https://github.com/PAIR-code/lit/commit/353b96ea5fd0aca9ace2ac47491b99d58cbbbc67),
      [ed07199](https://github.com/PAIR-code/lit/commit/ed07199189bce50446e05506cdfb8260781977eb),
      [184c8c6](https://github.com/PAIR-code/lit/commit/184c8c684c1f497f8911a5e886cec604b46c12f9),
      [14f82d5](https://github.com/PAIR-code/lit/commit/14f82d53b2e41b2cc088db3c1df3ebac5aee193a),
      [764674a](https://github.com/PAIR-code/lit/commit/764674a0430fc8e55535e09ad4bae4dc1eac1234)
* Changes to component `is_compatible()` signature
    * Added checks to some generators –
      [9b2de92](https://github.com/PAIR-code/lit/commit/9b2de92101b0a0c4961007a0a37fa936ee708e29),
      [db94849](https://github.com/PAIR-code/lit/commit/db948496d7b040463328ce926499d79e9a4d434d)
    * Added Dataset parameter to all checks –
      [ecd3a66](https://github.com/PAIR-code/lit/commit/ecd3a6623f2a0d45ae26c74d0d72fb68b7bcb9aa)
* Adds `core` components library to encapsulate default interpreters,
  generators, and metrics –
  [9ea4ab2](https://github.com/PAIR-code/lit/commit/9ea4ab264f6d9b03ee19ab8af4309e97862c089a)
* Removed the Color module –
  [b18d887](https://github.com/PAIR-code/lit/commit/b18d8871ea7ab1d2b5e4c671d33653d32f87d952)
* Removed the Slice module –
  [7db22ae](https://github.com/PAIR-code/lit/commit/7db22ae197650935ab916b248ca3c06f8593afb5)
* Moved star button to Data Table module –
  [cd14f35](https://github.com/PAIR-code/lit/commit/cd14f355781500b07a433a5df58d2ca0ec8ed6f8)
* Salience Maps now inside of expansion panels with popup controls –
  [1994425](https://github.com/PAIR-code/lit/commit/199442552586fa48780a33166cd6927ba4ab3530)
* Metrics
    * Promotion to a major `component` type –
      [de7d8ba](https://github.com/PAIR-code/lit/commit/de7d8ba26e74ecf2fd8a7700352e0d6d469d22ac)
    * Improved compatibility checks –
      [0d8341d](https://github.com/PAIR-code/lit/commit/0d8341d9f120359bec86c983c5618dd59bb6f591)

### New Stuff

* Common Color Legend element –
  [f846772](https://github.com/PAIR-code/lit/commit/f8467720d33dd8ef3d0da5c5a12eed2db37bb4b0),
  [7a1e26a](https://github.com/PAIR-code/lit/commit/7a1e26a9759882e0bf697363298e68f969c24a84),
  [0cc934c](https://github.com/PAIR-code/lit/commit/0cc934c8980a6d4563319087fd7e9ee5201acd04)
* Common Expansion Panel element –
  [2d67ce](https://github.com/PAIR-code/lit/commit/2d670ce70a6e41d7c2fc1d4d9b8c37c2b3b8876b)
* Common Faceting Control –
  [0f46e16](https://github.com/PAIR-code/lit/commit/0f46e166595c83773611a715be694100d89cace0),
  [b109f9b](https://github.com/PAIR-code/lit/commit/b109f9b8cad9c26f328c1634122fe874309d5b53),
  [8993f9b](https://github.com/PAIR-code/lit/commit/8993f9b5cd92f0d4fdfcd1c9e654c2aa4e15fb98),
  [670abeb](https://github.com/PAIR-code/lit/commit/670abeb25dbdc747067fae725a50a873355eb368)
* Common Popup element –
  [1994425](https://github.com/PAIR-code/lit/commit/199442552586fa48780a33166cd6927ba4ab3530),
  [cca3511](https://github.com/PAIR-code/lit/commit/cca3511322189ddb49bb6a533576d01f532a6f23)
* A new Dive module for exploring your data –
  [155e0c4](https://github.com/PAIR-code/lit/commit/155e0c4f1fb8198a18186c432bdb1516e9910f9e),
  [1d17ca2](https://github.com/PAIR-code/lit/commit/1d17ca23245765d2ded6790902eb5c4b9af3c954),
  [a0da9cf](https://github.com/PAIR-code/lit/commit/a0da9cf0643c2468b06d964b942aa523cd06069c)
* Copy or download data from Table elements –
  [d23ecfc](https://github.com/PAIR-code/lit/commit/d23ecfc74993dc932d88e412170cbb3cf6998408)
* Training Data Attribution module –
  [5ff9102](https://github.com/PAIR-code/lit/commit/5ff91029b05bea2d47835b81e840387ce8e70294),
  [c7398f8](https://github.com/PAIR-code/lit/commit/c7398f82f845180192a76eba2c0caade05a5c0bc)
* Tabular Feature Attribution module with a heatmap mode and
  [SHAP](https://shap.readthedocs.io/en/latest/index.html) interpreter –
  [45e526c](https://github.com/PAIR-code/lit/commit/45e526c76c586ba3539f28c0e03ab4adb9825def),
  [76379ad](https://github.com/PAIR-code/lit/commit/76379adac37f7e284faf979673cbb0399a36d8ee)
* Salience Clustering module –
  [8f3c26c](https://github.com/PAIR-code/lit/commit/8f3c26c60b652ae22cbb8c64e4b2212747c40413),
  [fb795e8](https://github.com/PAIR-code/lit/commit/fb795e8949b4b430c96e6d02d001e0a9aedd6c42),
  [49faa00](https://github.com/PAIR-code/lit/commit/49faa002d648a4b128c862f63b8202bf739c75d2),
  [e35d8d8](https://github.com/PAIR-code/lit/commit/e35d8d84edb9bc1ced5c4bc5e7bbcd8307dc99ac),
  [7505861](https://github.com/PAIR-code/lit/commit/75058615bc46b53c82b8561ae2bf80ff4c0eb2aa),
  [f970958](https://github.com/PAIR-code/lit/commit/f970958024c821880b3238d7a2f293b947a4e1e7)
* Selection state syncing in Python notebooks –
  [08abc2c](https://github.com/PAIR-code/lit/commit/08abc2ca3a25f368823a4a9f3ba9d5b5ebeac7a6),
  [06613b9](https://github.com/PAIR-code/lit/commit/06613b909173978c1d4648c8b37c28269a783c14)
* Unified DataService –
  [9bdc23e](https://github.com/PAIR-code/lit/commit/9bdc23e7890e8afeb7ab6dcc89c8cb7730c10b26),
  [00749fc](https://github.com/PAIR-code/lit/commit/00749fc0d4f83cad204a69d792f602c63b1ff676)
* AUC ROC and AUC PR Curve interpreters and module  –
  [51842ba](https://github.com/PAIR-code/lit/commit/51842babef63f9aa29d1d2add14633c4640627fc),
  [0f9fd4d](https://github.com/PAIR-code/lit/commit/0f9fd4dccc9e6375c577012191f89c3fb7067b01),
  [0558ef5](https://github.com/PAIR-code/lit/commit/0558ef52276ed6797a7a6f9d88721a50b6d6a792),
  [4efd58e](https://github.com/PAIR-code/lit/commit/4efd58e788a3cd38852961b136d8461f3b75b3d7)
* Splash screen documentation –
  [1f09ae9](https://github.com/PAIR-code/lit/commit/1f09ae9ca326dbaf0e5541f0f24370b56bcc6d1b),
  [cfabe78](https://github.com/PAIR-code/lit/commit/cfabe7865df5fd51ff8c483296f3fccc0fa30d28),
  [aca35d8](https://github.com/PAIR-code/lit/commit/aca35d832ad00a4bf35fd27adf35ba76f4d0d87f)
* Added a `GeneratedURL` type that displays in the Generated Text module –
  [bb06368](https://github.com/PAIR-code/lit/commit/bb06368602cfcca656746525de16a603e2359cb3)
* Added new built-in
  [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) and Exact Match
  metrics –
  [6773927](https://github.com/PAIR-code/lit/commit/67739270434388a63627ee1bc405bc16923dd631),
  [eac9382](https://github.com/PAIR-code/lit/commit/eac9382cebbc9d1e974ec0e7b6bc1cd528a4df1a)
* Input Salience demo –
  [a98edce](https://github.com/PAIR-code/lit/commit/a98edce9caf8e8481f4105cb26b57d5d0429f963),
  [75ff835](https://github.com/PAIR-code/lit/commit/75ff835ed2e051899d3839ab0ca4360bbf0b9897),
  [55579de](https://github.com/PAIR-code/lit/commit/55579de33fd292b34fecbcd058686cab1f05fd74)
* Model and Dataset validation –
  [0fef77a](https://github.com/PAIR-code/lit/commit/0fef77a7835bbfc9a022a9b1c99b10fc9f5a55c7)
* Tutorials written by our Google Summer of Code contributor,
  [Anjishnu Mukherjee](https://github.com/iamshnoo)
    * Using LIT for Tabular Feature Attribution –
      [2c0703d](https://github.com/PAIR-code/lit/commit/2c0703d69b3c5d3f9ef5aa4c03fe3c6262e707c3)
    * Making Sense of Salience Maps –
      [4159527](https://github.com/PAIR-code/lit/commit/415952702893febbcea9d631ad1a289a3e43e27c)

### Non-breaking Changes, Bug Fixes, and Enhancements

* Added Dataset embeddings to Embeddings projector –
  [78e2e9c](https://github.com/PAIR-code/lit/commit/78e2e9c05c831fafd360da5f1c3b9b4e12054df9),
  [3c0929f](https://github.com/PAIR-code/lit/commit/3c0929f9bb293391471d5bc3c1219b6025946354),
  [e7ac98b](https://github.com/PAIR-code/lit/commit/e7ac98bbabb5b0bf40bd956724cc5a63aef10350)
* Added a “sparse” mode to Classification Results –
  [20a8f31](https://github.com/PAIR-code/lit/commit/20a8f316ec0b3d68cd131b785b8dfd6fa61ab3e5)
* Added “Show only generated” option to Data Table module –
  [4851c9d](https://github.com/PAIR-code/lit/commit/4851c9de8917d35e2e1cc66d8d33d52f78418acf)
* Added threshold property for `MulticlassPreds` that allows for default
  threshold values other than 0.5 –
  [5e91b19](https://github.com/PAIR-code/lit/commit/5e91b1984700f6c1bb25b05d25e091d8d522c7e9)
* Added toggle for module duplication direction –
  [4e05a75](https://github.com/PAIR-code/lit/commit/4e05a759bca13afe857abd10abfdb5229d1ae622)
* Clickable links in the Generated Images module –
  [8cf8119](https://github.com/PAIR-code/lit/commit/8cf8119cbdaa5beea2b615d2eadb66630234af38)
* Constructor parameters for salience interpreters – [
  ab057b5](https://github.com/PAIR-code/lit/commit/ab057b55a938a59b87f08597050af5adfa2b8bcc)
* Image upload in Datapoint Editor –
  [a23b146](https://github.com/PAIR-code/lit/commit/a23b14676c7cb4fa7b82e42f9b6c036108801a54)
* Markdown support in LIT component descriptions –
  [0eaa00c](https://github.com/PAIR-code/lit/commit/0eaa00c1f58097c6e77354678f0b603eeabe74cd)
* Selection updates based on interactions in Metrics module –
  [c3b6a0c](https://github.com/PAIR-code/lit/commit/c3b6a0cceb300de5e18ff9bd68cf8c29b49b49b8)
* Support for many. new types of inputs in the Datapoint editor, including
  `GeneratedText`, `GeneratedTextCandidates`, `MultiSegmentAnnotation`,
  `Tokens`, `SparseMultilabel`, and `SparseMultilabelPreds`
* Various styling fixes and code cleanup efforts
* Docs, FAQ, and README updates

## Release 0.4.1

This is a bug fix release aimed at improving visual clarity and common
workflows.

The UI has been slightly revamped, bugs have been fixed, and new capabilities
have been added. Notable changes include:

- Adds "open in new tab" feature to LIT Notebook widget
- Adds support for `SparseMultilabelPreds` to LIME
- Improves color consistency across the UI
- Switching NumPy instead of SciKit Learn for PCA
- Ensuring all built-in demos are compatible with the Docker
- Updating the Dockerfile to support run-time `DEMO_NAME` and `DEMO_PORT` args
- Fixed a rendering bug in the Confusion Matrix related column and row spans
  when "hide empty labels" is turned on

## Release 0.4

This release adds a lot of new features. The website and documentation have
been updated accordingly.

The UI has been slightly revamped, bugs have been fixed, and new capabilities
have been added. Notable changes include:
- Support for Google Cloud Vertex AI notebooks.
- Preliminary support for tabular and image data, in addition to NLP models.
- Addition of TCAV global interpretability method.
- New counterfactual generators for ablating or flipping text tokens for
  minimal changes to flip predictions.
- New counterfactual generator for tabular data for minimal changes to flip
  predictions.
- Partial dependence plots for tabular input features.
- Ability to set binary classification thresholds separately for different
  facets of the dataset
- Controls to find optimal thresholds across facets given different fairness
  constraints, such as demographic parity or equal opportunity.

## Release 0.3

This release adds the ability to use LIT directly in colab and jupyter
notebooks. The website and documentation have been updated accordingly.

The UI has been slightly revamped, bugs have been fixed, and new capabilities
have been added. Notable changes include:
- Notebook mode added.
- New annotated text visualization module added.
- Allow saving/loading of generated datapoints, and dynamic adding of new
  datasets by path in the UI.
- Added synchronized scrolling between duplicated modules when comparing
  datapoints or models.
- Added a focus service for visually linking focus (i.e. hover) states between
  components.
- Allow layouts to be specified on LIT creation in python.

## Release 0.2

This release of LIT coincides with the EMNLP 2020 conference, where the LIT
paper was presented, and the publication of the LIT website, including tutorials
and hosted demos.

The UI has been slightly revamped, bugs have been fixed, and new capabilities
have been added.

## Release 0.1.1

This release of LIT adds a pip package for easy installation, cleans up some of
the code and documentation, and adds more examples.

## Release 0.1

This is the initial release of LIT.
