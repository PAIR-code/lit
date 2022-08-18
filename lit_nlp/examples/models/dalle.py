"""Dalle model based on https://github.com/borisdayma/dalle-mini."""

from typing import Optional

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import image_utils
import time

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
import flax.training.common_utils

DalleBart = dalle_mini.DalleBart
DalleBartProcessor = dalle_mini.DalleBartProcessor
VQModel = vqgan_jax.modeling_flax_vqgan.VQModel
CLIPProcessor = transformers.CLIPProcessor
FlaxCLIPModel = transformers.FlaxCLIPModel
shard_prng_key = flax.training.common_utils.shard_prng_key
trange = tqdm.notebook.trange
shard = flax.training.common_utils.shard
replicate = flax.jax_utils.replicate
JsonDict = lit_types.JsonDict

class DalleModel(lit_model.Model):
  """Image to Text Model"""

  def __init__(self,
               model_name:str,
               predictions:int):
    super().__init__()

    # small model -> dalle-mini/dalle-mini/mini-1:v0  
    # larger one:"dalle-mini/dalle-mini/mega-1-fp16:latest"
    self.model = model_name
    self.n_predictions = predictions
    
    # Load Models
    # VQGAN model   
    vqgan_repo = "dalle-mini/vqgan_imagenet_f16_16384"
    vqgan_commit_id = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
    self.dalle_commit_id = None

    # load model
    self.dalle_bert_model, params = DalleBart.from_pretrained(
        self.model, revision=self.dalle_commit_id, dtype=jnp.float16, _do_init=False
    )
    self.vqgan, vqgan_params = VQModel.from_pretrained(
        vqgan_repo, revision=vqgan_commit_id, _do_init=False
    )
    
    self.params = replicate(params)
    self.vqgan_params = replicate(vqgan_params)

    # Another CLIP model to generate CLIP score
    clip_repo = "openai/clip-vit-base-patch32"
    clip_commit_id = None

    # Load CLIP
    self.clip, clip_params = FlaxCLIPModel.from_pretrained(
        clip_repo, revision=clip_commit_id, dtype=jnp.float16, _do_init=False
    )

    self.clip_processor = CLIPProcessor.from_pretrained(clip_repo, revision=clip_commit_id)
    self.clip_params = replicate(clip_params)

    
  # LIT API implementation
  def max_minibatch_size(self) -> int:
    return 8

  def predict_minibatch(self, inputs):
    """Model prediction based on code pipeline in doc https://github.com/borisdayma/dalle-mini"""

     # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        print(tokenized_prompt)
        return self.dalle_bert_model.generate(
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
        return self.vqgan.decode_code(indices, params=params)
    
    # score images
    @partial(jax.pmap, axis_name="batch")
    def p_clip(inputs, params):
        logits = self.clip(params=params, **inputs).logits_per_image
        return logits

    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    
    prompts = [ex["prompt"] for ex in inputs]

    processor = DalleBartProcessor.from_pretrained(self.model, revision=self.dalle_commit_id)
    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)
    
    # We can customize generation parameters ( https://huggingface.co/blog/how-to-generate)
    gen_top_k:Optional[int]  = None
    gen_top_p:Optional[float]= None
    temperature:Optional[float] = None
    cond_scale:Optional[float] = 10.0
  
    # generate images
    images = []
    pil_images = []
    for i in trange(max(self.n_predictions // jax.device_count(), 1)):
        start = time.process_time()

        # get a new key
        # as per documentation Keys are passed to the model on each device to generate unique inference.
        # if key will be same than it will generate same images
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            self.params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, self.vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            # need pil format images too to generate CLIP score
            pil_images.append(img)
            image_str = image_utils.convert_pil_to_image_str(img)
            images.append(image_str)
            # Output image process time
            print(f'Image time: {time.process_time() - start}')

    # get CLIP scores
    clip_inputs = self.clip_processor(
        text=prompts * jax.device_count(),
        images=pil_images,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).data
    logits = p_clip(shard(clip_inputs), self.clip_params)

    # organize images and CLIP score per prompt
    p = len(prompts)
    logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()
    final_images =[]
    clip_score = []
    images_per_prompt = []
    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt}\n")
        if len(logits[i].argsort()[::-1]) == 1:
            my_loop = [0]*self.n_predictions
        else:
            my_loop = logits[i].argsort()[::-1]    
        for idx in range(len(my_loop)):
            images_per_prompt.append(images[idx * p + i])
            # Genereates CLIP score it needs more than one prompt to run hense the condition
            # print(f"Score: {jnp.asarray(logits[i][idx], dtype=jnp.float32):.2f}\n")
            if len(np.shape(logits)) == 1:
                clip_score.append((str(logits[idx]), None))
            else:
                clip_score.append((str(logits[i][idx]), None))
        final_images.append({
            'image': images_per_prompt,
            'clip_score': clip_score
            })
        images_per_prompt = []
        clip_score = []

    # return ImageBytes Array
    return final_images
    
# {'image': ['data:image/png;base64,']}]
  def input_spec(self):
    return {
        "prompt": lit_types.TextSegment(),
    }

  def output_spec(self):
    # returns only one image for now
    return {
        "image": lit_types.ImageBytesList(),
        "clip_score": lit_types.GeneratedTextCandidates(parent="prompt"),
    }
  
# ]}, {'image': ['data:image/png;base64, -> # {'image': ['data:image/png;base64','data:image/png;base64'