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

import numpy as np
import torch
import tqdm
import warnings
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# ##############
#
# Image Loading (InternVL Official Way)
#
# ##############

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

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height), Image.LANCZOS)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size), Image.LANCZOS)
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    elif isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    else:
        image = image_file
        
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ##############
#
# Hooks
#
# ##############

def set_hs_patch_hooks_internvl(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = hs_

    def post_hook(module, input, output):
      if "skip_ln" in name or "mlp" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name or "mlp" in name:
          output[0][position_] = hs_
        else:
          output[0][0, position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook
  
  hooks = []
  for i in hs_patch_config:
      patch_hook = patch_hs(
          f"patch_{module}_{i}",
          position_hs=hs_patch_config[i],
          patch_input=patch_input,
          generation_mode=generation_mode,
      )

      if patch_input:
          if module == "hs":
              hooks.append(
                  model.language_model.model.layers[i].register_forward_pre_hook(patch_hook)
              )
          elif module == "mlp":
              hooks.append(
                  model.language_model.model.layers[i].mlp.register_forward_pre_hook(patch_hook)
              )
          elif module == "attn":
              hooks.append(
                  model.language_model.model.layers[i].self_attn.register_forward_pre_hook(patch_hook)
              )
          else:
              raise ValueError("Module %s not supported" % module)
      else:
          if skip_final_ln and i == len(model.language_model.model.layers) - 1 and module == "hs":
              hooks.append(
                  model.language_model.model.norm.register_forward_hook(
                      patch_hs(
                          f"patch_hs_{i}_skip_ln",
                          hs_patch_config[i],
                          patch_input,
                          generation_mode,
                      )
                  )
              )
          else:
              if module == "hs":
                  hooks.append(
                      model.language_model.model.layers[i].register_forward_hook(patch_hook)
                  )
              elif module == "mlp":
                  hooks.append(
                      model.language_model.model.layers[i].mlp.register_forward_hook(patch_hook)
                  )
              elif module == "attn":
                  hooks.append(
                      model.language_model.model.layers[i].self_attn.register_forward_hook(patch_hook)
                  )
              else:
                  raise ValueError("Module %s not supported" % module)

  return hooks


def remove_hooks(hooks):
  for hook in hooks:
    hook.remove()


# ##############
#
# Inspection
#
# ##############

def find_image_end_index(mt, prompt):
    messages = [
        {
        "role": "user",
        "content": [
            {"type": "image", "url": "..."},
            {"type": "text", "text": prompt},
        ],
        }
    ]
    prompt_token_len = len(mt.tokenizer.apply_chat_template(messages)) + 2
    return (-1) * prompt_token_len



def inspect_vision_in_lm(
    mt,
    prompt_target,
    source_image,
    target_image,
    layer_source,
    layer_target,
    max_gen_len=20,
    verbose=False,
    temperature=None,
):
    """
    Inspect the vision features in the language model for InternVL.
    Uses the official InternVL loading and chat interface.
    """
    device = mt.device

    
    
    # Load images using InternVL's official method
    source_pixel_values = load_image(source_image, input_size=448, max_num=12).to(torch.bfloat16).to(device)
    target_pixel_values = load_image(target_image, input_size=448, max_num=12).to(torch.bfloat16).to(device)
    
    extracted_activations = {}

    def capture_hs(module, input, output):
        nonlocal extracted_activations
        if "image_features" not in extracted_activations:
            # output[0] is the hidden state tensor
            extracted_activations["image_features"] = output[0].clone()
            if verbose:
                print(f"Captured features from layer {layer_source}, shape: {output[0].shape}")

    # Register hook on source layer
    capture_hs_hook = mt.model.language_model.model.layers[layer_source].register_forward_hook(capture_hs)

    # Use model.chat for source (captures activations)
    # IMPORTANT: Use the SAME prompt as target to ensure token positions align!
    with torch.no_grad():
        try:
            # Suppress the pad_token_id warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*pad_token_id.*")
                
                # InternVL chat expects: (tokenizer, pixel_values, question, generation_config)
                _ = mt.model.chat(
                    mt.tokenizer,
                    source_pixel_values,
                    prompt_target,
                    generation_config=dict(
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=mt.tokenizer.eos_token_id
                    )
                )
        except Exception as e:
            print(f"Warning during source forward pass: {e}")
            import traceback
            traceback.print_exc()

    capture_hs_hook.remove()

    patched = False

    def patch_hs(module, input, output):
        nonlocal extracted_activations
        nonlocal patched

        img_start_idx = 39
        img_end_idx = 298

        output[0][0][img_start_idx:img_end_idx] = extracted_activations["image_features"][0][img_start_idx:img_end_idx]
        return output
        
        # if patched:
        #     if verbose:
        #         print("Already patched, skipping...")
        #     return output


        # if "image_features" in extracted_activations:
        #     for i in range(img_start_idx, img_end_idx):
        #         output[0][0][i] = extracted_activations["image_features"][0][i]
        
        
        # patched = True
        # return output
       

    patch_hook_handle = mt.model.language_model.model.layers[layer_target].register_forward_hook(patch_hs)
    
    # Generate with patched features
    with torch.no_grad():
        try:
            # Suppress the pad_token_id warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*pad_token_id.*")
                
                output_text = mt.model.chat(
                    mt.tokenizer,
                    target_pixel_values,
                    prompt_target,
                    generation_config=dict(
                        max_new_tokens=max_gen_len,
                        do_sample=(temperature is not None and temperature > 0),
                        temperature=temperature if temperature is not None else 1.0,
                        pad_token_id=mt.tokenizer.eos_token_id
                    )
                )
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            output_text = ""

    patch_hook_handle.remove()

    return output_text
