# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helpers for getting default values for LitApp configurations."""
from lit_nlp.api import components as lit_components
from lit_nlp.api import model as lit_model
from lit_nlp.components import ablation_flip
from lit_nlp.components import classification_results
from lit_nlp.components import curves
from lit_nlp.components import gradient_maps
from lit_nlp.components import hotflip
from lit_nlp.components import lime_explainer
from lit_nlp.components import metrics
from lit_nlp.components import model_salience
from lit_nlp.components import nearest_neighbors
from lit_nlp.components import pca
from lit_nlp.components import pdp
from lit_nlp.components import projection
from lit_nlp.components import regression_results
from lit_nlp.components import salience_clustering
from lit_nlp.components import scrambler
from lit_nlp.components import tcav
from lit_nlp.components import thresholder
from lit_nlp.components import word_replacer

# pylint: disable=g-import-not-at-top
# pytype: disable=import-error
try:
  from lit_nlp.components import shap_explainer

  _SHAP_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
  _SHAP_AVAILABLE = False

try:
  from lit_nlp.components import umap

  _UMAP_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
  _UMAP_AVAILABLE = False
# pylint: enable=g-import-not-at-top
# pytype: enable=import-error


def default_generators() -> dict[str, lit_components.Generator]:
  """Returns a dict of the default generators used in a LitApp."""
  return {
      'Ablation Flip': ablation_flip.AblationFlip(),
      'Hotflip': hotflip.HotFlip(),
      'Scrambler': scrambler.Scrambler(),
      'Word Replacer': word_replacer.WordReplacer(),
  }


def required_interpreters() -> dict[str, lit_components.Interpreter]:
  """Returns a dict of required interpreters.

  These are used by multiple core modules, and without them the frontend will
  likely throw errors.
  """
  # Ensure the prediction analysis interpreters are included.
  prediction_analysis_interpreters: dict[str, lit_components.Interpreter] = {
      'classification': classification_results.ClassificationInterpreter(),
      'regression': regression_results.RegressionInterpreter(),
  }
  return prediction_analysis_interpreters


def default_interpreters(
    models: dict[str, lit_model.Model]
) -> dict[str, lit_components.Interpreter]:
  """Returns a dict of the default interpreters used in a LitApp.

  Args:
    models: A dictionary of models that included in the LitApp that may provide
      their own salience information.
  """
  interpreters = required_interpreters()

  # Ensure the embedding-based interpreters are included.
  embedding_interpreters: dict[str, lit_components.Interpreter] = {
      'nearest neighbors': nearest_neighbors.NearestNeighbors(),
      # Embedding projectors expose a standard interface, but get special
      # handling so we can precompute the projections if requested.
      'pca': projection.ProjectionManager(pca.PCAModel),
  }

  if _UMAP_AVAILABLE:
    embedding_interpreters['umap'] = projection.ProjectionManager(
        umap.UmapModel
    )

  gradient_map_interpreters: dict[str, lit_components.Interpreter] = {
      'Grad L2 Norm': gradient_maps.GradientNorm(),
      'Grad ⋅ Input': gradient_maps.GradientDotInput(),
      'Integrated Gradients': gradient_maps.IntegratedGradients(),
      'LIME': lime_explainer.LIME(),
  }

  # pyformat: disable
  core_interpreters: dict[str, lit_components.Interpreter] = {
      'Model-provided salience': model_salience.ModelSalience(models),
      'tcav': tcav.TCAV(),
      'curves': curves.CurvesInterpreter(),
      'thresholder': thresholder.Thresholder(),
      'pdp': pdp.PdpInterpreter(),
      'Salience Clustering': salience_clustering.SalienceClustering(
          dict(gradient_map_interpreters)
      ),
  }
  # pyformat: enable

  if _SHAP_AVAILABLE:
    core_interpreters['Tabular SHAP'] = shap_explainer.TabularShapExplainer()

  interpreters.update(
      **core_interpreters, **gradient_map_interpreters, **embedding_interpreters
  )
  return interpreters


# TODO(b/254833485): Update typing to be a dict[str, lit_components.Metrics]
# once the Wrapper classes in metrics.py inherit from lit_components.Metrics.
def default_metrics() -> dict[str, lit_components.Interpreter]:
  return {
      'regression': metrics.RegressionMetrics(),
      'multiclass': metrics.MulticlassMetrics(),
      'multilabel': metrics.MultilabelMetrics(),
      'paired': metrics.MulticlassPairedMetrics(),
      'bleu': metrics.CorpusBLEU(),
      'rouge': metrics.RougeL(),
      'exactmatch': metrics.ExactMatchMetrics(),
  }
