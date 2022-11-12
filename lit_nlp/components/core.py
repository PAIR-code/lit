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
from typing import Union
from lit_nlp.api import components as lit_components
from lit_nlp.api import model as lit_model
from lit_nlp.components import ablation_flip
from lit_nlp.components import classification_results
from lit_nlp.components import curves
from lit_nlp.components import gradient_maps
from lit_nlp.components import hotflip
from lit_nlp.components import lemon_explainer
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
from lit_nlp.components import shap_explainer
from lit_nlp.components import tcav
from lit_nlp.components import thresholder
from lit_nlp.components import umap
from lit_nlp.components import word_replacer

ComponentGroup = lit_components.ComponentGroup
Generator = lit_components.Generator
Interpreter = lit_components.Interpreter
Model = lit_model.Model


def default_generators() -> dict[str, Generator]:
  """Returns a dict of the default generators used in a LitApp."""
  return {
      'Ablation Flip': ablation_flip.AblationFlip(),
      'Hotflip': hotflip.HotFlip(),
      'Scrambler': scrambler.Scrambler(),
      'Word Replacer': word_replacer.WordReplacer(),
  }


def default_interpreters(models: dict[str, Model]) -> dict[str, Interpreter]:
  """Returns a dict of the default interpreters (and metrics) used in a LitApp.

  Args:
    models: A dictionary of models that included in the LitApp that may provide
      thier own salience information.
  """
  # Ensure the embedding-based interpreters are included.
  embedding_based_interpreters: dict[str, Interpreter] = {
      'nearest neighbors': nearest_neighbors.NearestNeighbors(),
      # Embedding projectors expose a standard interface, but get special
      # handling so we can precompute the projections if requested.
      'pca': projection.ProjectionManager(pca.PCAModel),
      'umap': projection.ProjectionManager(umap.UmapModel),
  }
  gradient_map_interpreters: dict[str, Interpreter] = {
      'Grad L2 Norm': gradient_maps.GradientNorm(),
      'Grad ⋅ Input': gradient_maps.GradientDotInput(),
      'Integrated Gradients': gradient_maps.IntegratedGradients(),
      'LIME': lime_explainer.LIME(),
  }
  # Ensure the prediction analysis interpreters are included.
  prediction_analysis_interpreters: dict[str, Interpreter] = {
      'classification': classification_results.ClassificationInterpreter(),
      'regression': regression_results.RegressionInterpreter(),
  }
  # pyformat: disable
  interpreters: dict[str, Union[ComponentGroup, Interpreter]] = {
      'Model-provided salience': model_salience.ModelSalience(models),
      'counterfactual explainer': lemon_explainer.LEMON(),
      'tcav': tcav.TCAV(),
      'curves': curves.CurvesInterpreter(),
      'thresholder': thresholder.Thresholder(),
      'metrics': default_metrics(),
      'pdp': pdp.PdpInterpreter(),
      'Salience Clustering': salience_clustering.SalienceClustering(
          gradient_map_interpreters),
      'Tabular SHAP': shap_explainer.TabularShapExplainer(),
  }
  # pyformat: enable
  interpreters.update(**gradient_map_interpreters,
                      **prediction_analysis_interpreters,
                      **embedding_based_interpreters)
  return interpreters


def default_metrics() -> ComponentGroup:
  return ComponentGroup({
      'regression': metrics.RegressionMetrics(),
      'multiclass': metrics.MulticlassMetrics(),
      'paired': metrics.MulticlassPairedMetrics(),
      'bleu': metrics.CorpusBLEU(),
      'rouge': metrics.RougeL(),
      'exactmatch': metrics.ExactMatchMetrics(),
  })
