"""Dalle model based on https://github.com/borisdayma/dalle-mini."""
from ast import Str
import re
from typing import List

import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import image_utils
from PIL import Image
import numpy as np

import jax
import  jax.numpy as jnp
from functools import partial
import random
# Load models & tokenizer
import dalle_mini 
import vqgan_jax
import vqgan_jax.modeling_flax_vqgan
import transformers
import flax
import flax.training.common_utils
import tqdm.notebook


DalleBart = dalle_mini.DalleBart
DalleBartProcessor = dalle_mini.DalleBartProcessor
VQModel = vqgan_jax.modeling_flax_vqgan.VQModel
CLIPProcessor = transformers.CLIPProcessor
FlaxCLIPModel = transformers.FlaxCLIPModel
shard_prng_key = flax.training.common_utils.shard_prng_key
trange = tqdm.notebook.trange

replicate = flax.jax_utils.replicate
JsonDict = lit_types.JsonDict

class DalleModel(lit_model.Model):
  """Image to Text Model"""

 

  def __init__(self,
              model_name:Str,
              predictions:int):
    super().__init__()

    self.model = model_name
    self.predictions = predictions
    
  ##
  # LIT API implementation
  def max_minibatch_size(self) -> int:
    return 8

  def predict_minibatch(self, inputs):
    """Model prediction based on code pipeline in doc https://github.com/borisdayma/dalle-mini"""
    # tokenize the text. -> then return prediction
    # Load VQGAN
    # VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    # small model -> dalle-mini/dalle-mini/mini-1:v0  larger one:"dalle-mini/dalle-mini/mega-1-fp16:latest"
    DALLE_MODEL = self.model
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
    n_predictions = self.predictions
    # We can customize generation parameters ( https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    # generate images
    images = []
    final_Output = []
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
            image_str = image_utils.convert_pil_to_image_str(img)

            output = {
                'image': image_str
            }
            # print(output)
            final_Output.append(output)
    
    return final_Output
    

  def input_spec(self):
    return {
        "prompt": lit_types.TextSegment(),
    }

  def output_spec(self):
    # we have GeneratedTextCandidates for multiple text output but for image IDK.. I checked doc
    # Maybe a loop based on number of self.predictions? So if n_predictions = 5 in that case 
    # it should return 5 ImageBytes... is this a good approach? 
    return {
        "image": lit_types.ImageBytes()
    }
    
  
