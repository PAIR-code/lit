"""Dalle model based on https://github.com/borisdayma/dalle-mini."""

from collections.abc import Iterable
import functools
import random
from typing import Optional

import dalle_mini
import flax
from flax.training import common_utils as flax_common_utils
import jax
from jax import numpy as jnp
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import image_utils
import numpy as np
from PIL import Image
import tqdm.notebook
import transformers
from vqgan_jax import modeling_flax_vqgan as vqgan_flax

# DalleBart, VQModel & CLIP to generate Score
_JsonDict = lit_types.JsonDict
_CLIPProcessor = transformers.CLIPProcessor
_DalleBart = dalle_mini.DalleBart
_DalleBartProcessor = dalle_mini.DalleBartProcessor
_FlaxCLIPModel = transformers.FlaxCLIPModel
_VQModel = vqgan_flax.VQModel

# Some other functions
_flax_replicate = flax.jax_utils.replicate
_flax_shard = flax_common_utils.shard
_flax_shard_prng_key = flax_common_utils.shard_prng_key
_trange = tqdm.notebook.trange


class DalleModel(lit_model.Model):
  """LIT model wrapper for the Dalle-Mini Text-to-Image model.

  The model wrapper consists of a few already known works connected in an
  interesting way to generate images from the text:

  * VQGAN: The generative image model.
  * BART: A sequence-to-sequence autoencoder used to reconstruct input text.
  * CLIP: A scoring model that assesses the alingment of an image and a prompt.

  The basic flow within this model wrapper's predict() function is:

  1.  The BART encoder is fed the prompt text.
  2.  The BART decoder is sampled multiple times to generate candidates.
  3.  Each candidate is passed to VQGAN, which generates images.
  4.  CLIP then scores each generated image against the prompt.
  """

  def __init__(self,
               model_name: str,
               predictions: int = 1,
               mode_revision: Optional[str] = None,
               vqgan_repo: str = "dalle-mini/vqgan_imagenet_f16_16384",
               vqgan_revision: str = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9",
               clip_repo: str = "openai/clip-vit-base-patch32",
               clip_revision: Optional[str] = None):
    super().__init__()

    self.model = model_name
    self.n_predictions = predictions

    # Load Dalle model
    self.dalle_bart_model, params = _DalleBart.from_pretrained(
        self.model, revision=mode_revision, dtype=jnp.float16, _do_init=False
    )
    self.processor = _DalleBartProcessor.from_pretrained(
        self.model, revision=mode_revision
    )

    # Load the VQGan model
    self.vqgan, vqgan_params = _VQModel.from_pretrained(
        vqgan_repo, revision=vqgan_revision, _do_init=False
    )
    self.params = _flax_replicate(params)
    self.vqgan_params = _flax_replicate(vqgan_params)

    # Load CLIP model to generate CLIP score
    # Scores how accurate generated image is
    self.clip, clip_params = _FlaxCLIPModel.from_pretrained(
        clip_repo, revision=clip_revision, dtype=jnp.float16, _do_init=False
    )
    self.clip_processor = _CLIPProcessor.from_pretrained(
        clip_repo, revision=clip_revision
    )
    self.clip_params = _flax_replicate(clip_params)

  # LIT API implementation
  def max_minibatch_size(self) -> int:
    return 8

  def predict(
      self, inputs: Iterable[_JsonDict], **unused_kw
  ) -> Iterable[_JsonDict]:
    # Model prediction based on code pipeline in doc
    # https://github.com/borisdayma/dalle-mini
    #
    # Three models are required for prediction and it's in the following stages:
    #
    # 1.  First the prompt(text from which image is generated) is fed into
    #     BART encoder model (dalle_bart_model) then the BART decoder is sampled
    #     multiple times to generate candidates.
    #
    # 2.  Each candidate is passed to VQGAN which generates images. The
    #     generated images are stored in two arrays. One images[], which
    #     contains images in ImageBytes format second pil_images[] which
    #     contains the same images in PIL format which is used to generate CLIP
    #     score using the CLIP model.
    #
    # 3.  CLIP is another model by openai, which takes in an image and a prompt
    #     and tells how well they match by generating a score. At the last, the
    #     final output is formated such that it includes the CLIP score along
    #     with the generated images.

    # BART model inference function
    @functools.partial(jax.pmap, axis_name="batch")
    def p_generate(
        tokenized_prompt,
        key,
        params,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        condition_scale: float = 10.0
    ):
      return self.dalle_bart_model.generate(
          **tokenized_prompt,
          prng_key=key,
          params=params,
          top_k=top_k,
          top_p=top_p,
          temperature=temperature,
          condition_scale=condition_scale,
      )

    # VQGAN image decoder function
    @functools.partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
      return self.vqgan.decode_code(indices, params=params)

    # CLIP <image, prompt> scoring function
    @functools.partial(jax.pmap, axis_name="batch")
    def p_clip(inputs, params):
      logits = self.clip(params=params, **inputs).logits_per_image
      return logits

    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    prompts = [ex["prompt"] for ex in inputs]
    tokenized_prompts = self.processor(prompts)
    tokenized_prompt = _flax_replicate(tokenized_prompts)

    # images has all generated ImageByste
    # pil_images has images in pil format for clip score
    images = []
    pil_images = []
    for _ in _trange(max(self.n_predictions // jax.device_count(), 1)):
      # Get a new key; passed to the model on each device to generate unique
      # inference.
      key, subkey = jax.random.split(key)
      # generate images
      encoded_images = p_generate(
          tokenized_prompt, _flax_shard_prng_key(subkey), self.params
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
        # convert to ImageBytes
        image_str = image_utils.convert_pil_to_image_str(img)
        images.append(image_str)

    # get CLIP scores
    clip_inputs = self.clip_processor(
        text=prompts * jax.device_count(),
        images=pil_images,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).data

    # array containing clip score for all images
    p = len(prompts)
    logits = p_clip(_flax_shard(clip_inputs), self.clip_params)
    logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()

    # Organize VQGAN images and CLIP scores per prompt.
    #
    # Until now, images contains data structured as:
    # [
    #     prompt1_prediction_1,
    #     prompt2_prediction_1,
    #     prompt1_prediction_2,
    #     prompt2_prediction_2,
    # ]
    #
    # Below, the structure changes to...
    # [
    #     {'image': [prompt1_prediction_1, prompt1_prediction_2]},
    #     {'image': [prompt2_prediction_1, prompt2_prediction_2]},
    # ]
    #
    # ...and also adds clip score to check image accuracy
    final_images = []
    clip_score = []
    images_per_prompt = []

    # Add images as per prompt [prompt1_prediction_1,prompt1_prediction_2]
    # and also add clip_score
    for i in range(p):
      # logits shape is different if n_predictions is only 1, vs more than one
      clip_prompts = logits if logits.ndim == 0 else logits[i].argsort()[::-1]
      if logits.ndim == 0 or len(clip_prompts) == 1:
        my_loop = [0]*self.n_predictions
      else:
        my_loop = clip_prompts
      for idx in range(len(my_loop)):
        images_per_prompt.append(images[idx * p + i])
        # Append CLIP score depending on size hence the condition
        if logits.ndim == 0:
          clip_score.append((str(logits), None))
        elif len(np.shape(logits)) == 1:
          clip_score.append((str(logits[idx]), None))
        else:
          clip_score.append((str(logits[i][idx]), None))
      # Append to final list[JsonDict]
      final_images.append({
          "image": images_per_prompt,
          "clip_score": clip_score,
      })
      # Reset images & clip score for new prompt
      images_per_prompt = []
      clip_score = []

    return final_images

  def input_spec(self):
    return {"prompt": lit_types.TextSegment()}

  def output_spec(self):
    return {
        "image": lit_types.ImageBytesList(),
        "clip_score": lit_types.GeneratedTextCandidates(parent="prompt"),
    }
