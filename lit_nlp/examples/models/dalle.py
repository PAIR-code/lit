"""Dalle model based on https://github.com/borisdayma/dalle-mini."""

# Load models & tokenizer
import dalle_mini 
import transformers
import vqgan_jax
import vqgan_jax.modeling_flax_vqgan

# Import flax and other jax required modules
import flax
import flax.training.common_utils
from functools import partial
import jax
import  jax.numpy as jnp
import random

# Get LIT api & other modules
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import image_utils
import numpy as np
from PIL import Image
import tqdm.notebook
from typing import Optional

# DalleBart, VQModel & CLIP to generate Score
CLIPProcessor = transformers.CLIPProcessor
DalleBart = dalle_mini.DalleBart
DalleBartProcessor = dalle_mini.DalleBartProcessor
FlaxCLIPModel = transformers.FlaxCLIPModel
VQModel = vqgan_jax.modeling_flax_vqgan.VQModel

# Some other functions
JsonDict = lit_types.JsonDict
replicate = flax.jax_utils.replicate
shard = flax.training.common_utils.shard
shard_prng_key = flax.training.common_utils.shard_prng_key
trange = tqdm.notebook.trange




class DalleModel(lit_model.Model):
  """Text to Image Dalle Mini model

  The model consists of a few already known works connected in
  an interesting way to generate images from the text.

  This includes VQGAN, BART & CLIP.

  VQGAN is the generative image model, supposed to generate 
  new images. BART on the other hand is a sequence-to-sequence
  auto encoder used to reconstruct input text. 

  To understand why this class loads three different models need
  to understand how dalle-mini works:
  BART encoder is fed with the prompt(text from which image is generated).
  Then BART decoder is sampled multiple times to generate candidates.
  Each candidate is passed to VQGAN which generates images.

  CLIP is another model by openai, which takes in an image and a prompt 
  and tell how well they match by generating a score.
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
    self.dalle_bart_model, params = DalleBart.from_pretrained(
        self.model, revision=mode_revision, dtype=jnp.float16, _do_init=False
    )
    self.processor = DalleBartProcessor.from_pretrained(self.model, revision=mode_revision)

    # Load the VQGan model
    self.vqgan, vqgan_params = VQModel.from_pretrained(
        vqgan_repo, revision=vqgan_revision, _do_init=False
    )
    self.params = replicate(params)
    self.vqgan_params = replicate(vqgan_params)

    # Load CLIP model to generate CLIP score
    # Scores how accurate generated image is
    self.clip, clip_params = FlaxCLIPModel.from_pretrained(
        clip_repo, revision=clip_revision, dtype=jnp.float16, _do_init=False
    )
    self.clip_processor = CLIPProcessor.from_pretrained(clip_repo, revision=clip_revision)
    self.clip_params = replicate(clip_params)

    
  # LIT API implementation
  def max_minibatch_size(self) -> int:
    return 8

  def predict_minibatch(self, inputs):
    """Model prediction based on code pipeline in doc https://github.com/borisdayma/dalle-mini
    
    Three models are required for prediction and it's in the following stages:

    1) First the prompt(text from which image is generated) is fed into 
    BART encoder model (dalle_bart_model) then the BART decoder is sampled 
    multiple times to generate candidates.

    2) Each candidate is passed to VQGAN which generates images. The generated 
    images are stored in two arrays. One images[], which contains images in ImageBytes
    format second pil_images[] which contains the same images in PIL format which is used
    to generate CLIP score using the CLIP model.

    3) CLIP is another model by openai, which takes in an image and a prompt and tells 
    how well they match by generating a score. At the last, the final output is formated
    such that it includes the CLIP score along with the generated images.
    """

    # model inference
    @partial(jax.pmap, axis_name="batch")
    def p_generate(
        tokenized_prompt, key, params, top_k:Optional[int] = None, top_p:Optional[float] = None, 
        temperature:Optional[float] = None, condition_scale:float = 10.0
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

    tokenized_prompts = self.processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)
    
  
    # images has all generated ImageByste
    # pil_images has images in pil format for clip score
    images = []
    pil_images = []
    for i in trange(max(self.n_predictions // jax.device_count(), 1)):
        # get a new key
        # keys are passed to the model on each device to generate unique inference.
        # if key will be same than it won't be unique
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(tokenized_prompt,
                                    shard_prng_key(subkey),
                                    self.params)

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
    logits = p_clip(shard(clip_inputs), self.clip_params)
    logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()

    # organize images and CLIP score per prompt
    # till now our our images[] contains data like this: 
    # [prompt1_prediction_1, prompt2_prediction_1, prompt1_prediction_2, prompt2_prediction_2]
    # Below structure changes it to: [{'image': [prompt1_prediction_1,prompt1_prediction_2]}]
    # and also adds clip score to check image accuracy 
    final_images =[]
    clip_score = []
    images_per_prompt = []

    # Add images as per prompt [prompt1_prediction_1,prompt1_prediction_2]
    # and also add clip_score
    for i in range(p):
        # logits shape is different if n_predictions is only 1, vs more than one
        clip_prompts = logits if logits.ndim == 0 else logits[i].argsort()[::-1] 
        my_loop = [0]*self.n_predictions if logits.ndim == 0 or len(clip_prompts) == 1 else clip_prompts
        for idx in range(len(my_loop)):
            images_per_prompt.append(images[idx * p + i])
            # Append CLIP score depending on size hence the condition
            if logits.ndim == 0:
                clip_score.append((str(logits), None))
            elif len(np.shape(logits)) == 1:
                clip_score.append((str(logits[idx]), None))
            else:
                clip_score.append((str(logits[i][idx]), None))
        # Append to final List[Dict]
        final_images.append({
            "image": images_per_prompt,
            "clip_score": clip_score
        })
        # Reset images & clip score for new prompt
        images_per_prompt = []
        clip_score = []

    
    return final_images
    

  def input_spec(self):
    return {
        "prompt": lit_types.TextSegment(),
    }

  def output_spec(self):
    return {
        "image": lit_types.ImageBytesList(),
        "clip_score": lit_types.GeneratedTextCandidates(parent="prompt"),
    }
  
