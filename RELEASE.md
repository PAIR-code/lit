# Language Interpretability Tool releases

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
