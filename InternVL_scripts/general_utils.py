# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import math
import re
import torch
import transformers
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os

# Constants for image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height), Image.LANCZOS)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size), Image.LANCZOS)
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def prepare_inputs(prompt, image, tokenizer, device, max_num=12):
    """
    Prepare inputs for InternVL 3.5 model.
    InternVL 3.5 uses a special format where <image> placeholder gets replaced with IMG_CONTEXT tokens.
    """
    
    pixel_values = None
    if image is not None:
        pixel_values = load_image(image, max_num=max_num).to(torch.bfloat16).to(device)
        num_patches = pixel_values.size(0)
        
        # InternVL 3.5 uses IMG_CONTEXT token (id 151859) as placeholder
        # Each 448x448 patch generates 256 visual tokens
        # We need to create num_patches * 256 IMG_CONTEXT tokens
        
        # Get the IMG_CONTEXT token
        # InternVL 3.5 uses special token <|im_start|> and IMG_CONTEXT
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        # Check if tokenizer has this token
        if IMG_CONTEXT_TOKEN in tokenizer.get_vocab():
            img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        else:
            # Fallback: try to find it in additional_special_tokens
            img_context_token_id = 151859  # Default for InternVL 3.5
        
        # Create the image placeholder string
        # For InternVL 3.5: we need 256 IMG_CONTEXT tokens per patch
        num_img_tokens = num_patches * 256
        img_placeholder = IMG_CONTEXT_TOKEN * num_img_tokens
        
        # Format prompt with image placeholder
        if "<image>" in prompt:
            prompt = prompt.replace("<image>", img_placeholder)
        else:
            prompt = f"{img_placeholder}\n{prompt}"
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids).to(device),
    }
    
    if pixel_values is not None:
        inputs["pixel_values"] = pixel_values
        # image_flags: tensor indicating which image each patch belongs to
        # For single image: all 1s
        inputs["image_flags"] = torch.tensor([1] * num_patches, dtype=torch.long).to(device)

    return inputs


class ModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      processor=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      use_fast=True,
      device="cuda",
      ):
    if tokenizer is None:
      assert model_name is not None
      tokenizer = transformers.AutoTokenizer.from_pretrained(
          model_name, 
          use_fast=use_fast, 
          trust_remote_code=True
      )
    if model is None:
      assert model_name is not None
      # InternVL specific loading
      try:
          import flash_attn
          attn_impl = "flash_attention_2"
      except ImportError:
          attn_impl = "eager"
          
      # Device handling: If device_map is "auto", transformers handles splitting across GPUs.
      # If a specific device is passed, we might not want "auto".
      # However, for large models like InternVL 38B, we often need "auto".
      # The user requested "auto" device map and handling specific visible devices.
      
      device_map = "auto"
      
      model = transformers.AutoModel.from_pretrained(
          model_name, 
          low_cpu_mem_usage=low_cpu_mem_usage,
          torch_dtype=torch_dtype,
          trust_remote_code=True,
          attn_implementation=attn_impl,
          device_map=device_map
          )
      
      # If device_map is auto, model is already on devices. 
      # No need to move to 'device' manually unless it's cpu.
      if device == "cpu" or device_map is None:
          if device is not None:
              model = model.to(device)
      
      set_requires_grad(False, model)
      model.eval()
      
    self.tokenizer = tokenizer
    self.model = model
    self.device = device
    self.processor = processor
    # Attempt to auto-detect layers
    self.layer_names = []
    # For InternVL, the language model is usually in model.language_model
    if hasattr(model, "language_model"):
        lm = model.language_model
        self.layer_names = [
            n for n, _ in lm.named_modules()
            if (re.match(r"^(model)\.(layers)\.\d+$", n))
        ]
    
    if not self.layer_names:
        self.layer_names = [
            n
            for n, _ in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
    self.num_layers = len(self.layer_names)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )


def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)


def predict_from_input(model, inp):
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  p, preds = torch.max(probs, dim=1)
  return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)
