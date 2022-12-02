# Learning Interpretability Tool Release Notes

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
