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
from general_utils import decode_tokens, make_inputs, tokenizer_image_token, IMAGE_TOKEN_INDEX, prepare_inputs


# ##############
#
# Inspection
#
# ##############

def inspect_vision(
    mt,
    prompt_source,
    prompt_target,
    vision_layer_source,  # Layer in source vision transformer to extract from (0 to 31)
    generation_mode=False,
    max_gen_len=20,
    verbose=False,
    temperature=None,
    source_images=None,
    target_images=None,
    adv_tensor=None,
):
    """Inspection by patching source vision transformer layer output for Qwen2-VL model."""
    #########################################################
    # Prepare inputs for source and target prompts
    #########################################################
    # prepare inputs
    inp_source = prepare_inputs(prompt_source, source_images, mt.processor, mt.device)
    inp_target = prepare_inputs(prompt_target, target_images, mt.processor, mt.device)

    #########################################################
    # Capture hidden states from specified layer in source vision transformer
    #########################################################
    source_vision_hidden = None

    def capture_vision_layer(module, input, output):
        nonlocal source_vision_hidden
        source_vision_hidden = output

    # Register hook on the specified layer in the source vision transformer
    # Note: Qwen2-VL uses visual.blocks instead of vision_tower.vision_tower.vision_model.encoder.layers
    capture_hook = mt.model.visual.blocks[vision_layer_source].register_forward_hook(capture_vision_layer)

    # Run forward pass on source input to capture the hidden states
    with torch.no_grad():
        _ = mt.model(**inp_source)

    capture_hook.remove()

    # source_vision_hidden now contains the hidden states from the specified vision layer

    #########################################################
    # Patch the target vision output before merger (equivalent to before projection in LLaVA)
    #########################################################
    def replace_vision_output(module, input, output):
        # Replace the entire vision output with the source layer's hidden states
        if isinstance(source_vision_hidden, tuple):
            return source_vision_hidden[0]  # Take first element if tuple
        return source_vision_hidden

    # Register hook on the visual module's forward method (after all blocks, before merger which is equivalent to mm_projector)
    # In Qwen2-VL, the visual module output goes through the merger before being passed to the language model
    replace_hook = mt.model.visual.blocks[-1].register_forward_hook(replace_vision_output)

    #########################################################
    # Generate output with patched vision transformer output
    #########################################################
    if generation_mode:
        output_ids = mt.model.generate(**inp_target, max_new_tokens=128)
        output = mt.processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inp_target["input_ids"], output_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    else:
        output = mt.model(**inp_target)
        answer_prob, answer_t = torch.max(torch.softmax(output.logits[0, -1, :], dim=0), dim=0)
        output = (decode_tokens(mt.tokenizer, [answer_t])[0], round(answer_prob.cpu().item(), 4))

    if verbose:
        print(f"Prediction with vision transformer patched from source layer {vision_layer_source}:", output)

    # Clean up hook
    replace_hook.remove()

    return output

def inspect_vision_in_lm(
    mt,
    prompt_target,
    source_image,
    target_image,
    layer_source,
    layer_target,
    max_gen_len=20,
):
    """
    Inspect the vision features in the language model.
    """

    inp_source = prepare_inputs("", source_image.resize((384, 384)), mt.processor, mt.device)
    inp_target = prepare_inputs(prompt_target, target_image.resize((384, 384)), mt.processor, mt.device)

    img_start_idx = torch.where(inp_source["input_ids"][0] == 151652)[0][0]
    img_end_idx = torch.where(inp_source["input_ids"][0] == 151653)[0][0]

    extracted_activations = {}

    def capture_hs(module, input, output):
        nonlocal extracted_activations
        if "image_features" not in extracted_activations:
            extracted_activations["image_features"] = output[0].clone()

    capture_hs_hook = mt.model.model.layers[layer_source].register_forward_hook(capture_hs)

    with torch.no_grad():
        _ = mt.model(**inp_source)

    capture_hs_hook.remove()

    patched = False

    def patch_hs(module, input, output):
        nonlocal extracted_activations
        nonlocal patched
        if patched:
            return output
   
        for i in range(img_start_idx, img_end_idx):
            output[0][0][i] = extracted_activations["image_features"][0][i]


        patched = True
      

        return output

    patch_hook_handle = mt.model.model.layers[layer_target].register_forward_hook(patch_hs)
    
    with torch.no_grad():
        output_ids = mt.model.generate(**inp_target, max_new_tokens=max_gen_len)

    patch_hook_handle.remove()

    output = mt.processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inp_target["input_ids"], output_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    return output


