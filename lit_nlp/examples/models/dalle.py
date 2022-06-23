"""LIT wrappers for T5, supporting both HuggingFace and SavedModel formats."""
import re
from typing import List

import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import model_utils
from lit_nlp.lib import utils

import tensorflow as tf
import numpy as np
# tensorflow_text is required for T5 SavedModel
# import tensorflow_text  # pylint: disable=unused-import
import transformers

from rouge_score import rouge_scorer

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from functools import partial
import random
# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.training.common_utils import shard_prng_key
from IPython.display import display
import numpy as np
from PIL import Image
from tqdm.notebook import trange

BertTokenizer = transformers.BertTokenizer
FlaxBertForQuestionAnswering = transformers.FlaxBertForQuestionAnswering
JsonDict = lit_types.JsonDict

class DalleModel(lit_model.Model):
  """Question Answering Jax model based on TyDiQA Dataset ."""

 
  @property
  def max_seq_length(self):
    return self.model.config.max_position_embeddings

  def __init__(self, 
              model_name=None, 
              model=None,
              tokenizer=None,
              **config_kw):
    super().__init__()

    
    
  ##
  # LIT API implementation
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 8

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # tokenize the text. -> then return prediction
    # Load VQGAN
    # VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
    DALLE_COMMIT_ID = None
    model, params = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )
    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )
    
    params = replicate(params)
    vqgan_params = replicate(vqgan_params)

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)
    
    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    
    prompts = [ex["prompt"] for ex in inputs]

    processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)

    # generate Images
    # number of predictions per prompt
    n_predictions = 8

    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    print(f"Prompts: {prompts}\n")
    # generate images
    images = []
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
            print('Display->')
            display(img)
            print('  empty  ')
            print()
            print('Images array->')
            print(images)
    return images
    

  def input_spec(self):
    return {
        "prompt": lit_types.TextSegment(),
    }

  def output_spec(self):
    ret = {
        "image": lit_types.ImageBytes()
    }
    # Add attention and embeddings from each layer.
    # for i in range(self.model.config.num_hidden_layers):
    #   ret[f"layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
    #       align_in="tokens", align_out="tokens")
    #   ret[f"layer_{i:d}_avg_embedding"] = lit_types.Embeddings()
    return ret
  
