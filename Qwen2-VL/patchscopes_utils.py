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
# Hooks
#
# ##############

def set_hs_patch_hooks_qwen2vl(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, inputs):
            # (batch, sequence, hidden_state)
            input_len = len(inputs[0][0])
            if generation_mode and input_len == 1:
                return
            for position_, hs_ in position_hs:
                inputs[0][0, position_] = hs_

        def post_hook(module, inputs, output):
            # output 可能是 tuple 或 tensor，因此要依照實際形狀做判斷
            if "skip_ln" in name or "mlp" in name:
                # output: (batch, sequence, hidden_state)
                out_len = len(output[0])
                if generation_mode and out_len == 1:
                    return
                for position_, hs_ in position_hs:
                    output[0][position_] = hs_
            else:
                # output[0]: (batch, sequence, hidden_state)
                out_len = len(output[0][0])
                if generation_mode and out_len == 1:
                    return
                for position_, hs_ in position_hs:
                    output[0][0, position_] = hs_

        if patch_input:
            return pre_hook
        else:
            return post_hook

    hooks = []
    for i in hs_patch_config:
        patch_hook = patch_hs(
            f"patch_{module}_{i}",
            hs_patch_config[i],
            patch_input,
            generation_mode,
        )
        if patch_input:
            if module == "hs":
                hooks.append(
                    model.model.layers[i].register_forward_pre_hook(patch_hook)
                )
            elif module == "mlp":
                hooks.append(
                    model.model.layers[i].mlp.register_forward_pre_hook(patch_hook)
                )
            elif module == "attn":
                hooks.append(
                    model.model.layers[i].self_attn.register_forward_pre_hook(patch_hook)
                )
            else:
                raise ValueError("Module %s not supported" % module)
        else:
            if skip_final_ln and i == len(model.model.layers) - 1 and module == "hs":
                hooks.append(
                    model.model.norm.register_forward_hook(
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
                        model.model.layers[i].register_forward_hook(patch_hook)
                    )
                elif module == "mlp":
                    hooks.append(
                        model.model.layers[i].mlp.register_forward_hook(patch_hook)
                    )
                elif module == "attn":
                    hooks.append(
                        model.model.layers[i].self_attn.register_forward_hook(patch_hook)
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

def inspect(
    mt,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    generation_mode=False,
    max_gen_len=20,
    verbose=False,
    temperature=None,
):
  """Inspection via patching."""
  # Get the device from the model
  device = next(mt.model.parameters()).device

  # adjust position_target to be absolute rather than relative
  inp_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
  if position_target < 0:
    position_target = len(inp_target["input_ids"][0]) + position_target

  # first run the the model on prompt_patch and get all hidden states.
  inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
  if verbose:
    print(
        "prompt_patch:",
        [mt.tokenizer.decode(x) for x in inp_source["input_ids"][0]],
    )

  hs_cache_ = []
  # We manually store intermediate states that the model API does not expose
  store_hooks = []

  if hasattr(mt.model, 'language_model'):
    if module == "mlp":
        def store_mlp_hook(module, input, output):
            hs_cache_.append(output[0])

        for layer in mt.model.language_model.model.layers:
            store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))
    elif module == "attn":
        def store_attn_hook(module, input, output):
            hs_cache_.append(output[0].squeeze())

        for layer in mt.model.language_model.model.layers:
            store_hooks.append(layer.self_attn.register_forward_hook(store_attn_hook))

    # Call the model with the prepared inputs
    output = mt.model.language_model(**inp_source, output_hidden_states=True)
    
    if module == "hs":
        hs_cache_ = [
            output["hidden_states"][layer + 1][0] for layer in range(len(mt.model.language_model.model.layers))
        ]
  else:
    if module == "mlp":
      def store_mlp_hook(module, input, output):
        hs_cache_.append(output[0])

      for layer in mt.model.model.layers:
        store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))
    elif module == "attn":

      def store_attn_hook(module, input, output):
        hs_cache_.append(output[0].squeeze())

      for layer in mt.model.model.layers:
        store_hooks.append(layer.self_attn.register_forward_hook(store_attn_hook))

    output = mt.model(**inp_source, output_hidden_states=True)

    if module == "hs":
      hs_cache_ = [
          output["hidden_states"][layer + 1][0] for layer in range(mt.num_layers)
      ]

  remove_hooks(store_hooks)
  # now do a second run on prompt, while patching
  # a specific hidden state from the first run.
  hs_patch_config = {
      layer_target: [(
          position_target,
          hs_cache_[layer_source][position_source],
      )]
  }

  if layer_source == layer_target == mt.num_layers - 1:
    skip_final_ln = True
  else:
    skip_final_ln = False

  patch_hooks = mt.set_hs_patch_hooks(
      mt.model,
      hs_patch_config,
      module=module,
      patch_input=False,
      skip_final_ln=skip_final_ln,
      generation_mode=True,
  )

  inp_target = {k: v.to(device) for k, v in inp_target.items()}

  # Single prediction / generation
  if verbose:
    print(
        "prompt:", [mt.tokenizer.decode(x) for x in inp_source["input_ids"][0]]
    )
    print(
        f"patching position {position_target} with the hidden state from layer"
        f" {layer_source} at position {position_source}."
    )
  if generation_mode:
    # Checking if should perform temperature sampling, to allow smoother
    if hasattr(mt.model, 'language_model'):
      if temperature:
          output_toks = mt.model.language_model.generate(
              inp_target["input_ids"],
              max_length=len(inp_target["input_ids"][0]) + max_gen_len,
              pad_token_id=mt.tokenizer.eos_token_id,  # Use tokenizer's eos_token_id
              temperature=temperature,
              do_sample=True,
              top_k=0,
          )[0][len(inp_target["input_ids"][0]):]
      else:
          output_toks = mt.model.language_model.generate(
              inp_target["input_ids"],
              max_length=len(inp_target["input_ids"][0]) + max_gen_len,
              pad_token_id=mt.tokenizer.eos_token_id,  # Use tokenizer's eos_token_id
          )[0][len(inp_target["input_ids"][0]):]
    else:
      # non-repeating long outputs.
      if temperature:
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            max_length=len(inp_target["input_ids"][0]) + max_gen_len,
            pad_token_id=mt.model.generation_config.eos_token_id,
            temperature=temperature,
            do_sample=True,
            top_k=0,
        )[0][len(inp_target["input_ids"][0]) :]
      else:
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            max_length=len(inp_target["input_ids"][0]) + max_gen_len,
            pad_token_id=mt.model.generation_config.eos_token_id,
        )[0][len(inp_target["input_ids"][0]) :]

    output = mt.tokenizer.decode(output_toks)
    if verbose:
      print(
          "generation with patching: ",
          [mt.tokenizer.decode(x) for x in output_toks],
      )
  else:
    output = mt.model(**inp_target)
    answer_prob, answer_t = torch.max(
        torch.softmax(output.logits[0, -1, :], dim=0), dim=0
    )
    output = decode_tokens(mt.tokenizer, [answer_t])[0], round(
        answer_prob.cpu().item(), 4
    )
    if verbose:
      print("prediction with patching: ", output)

  # remove patching hooks
  remove_hooks(patch_hooks)

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

    capture_hs_hook = mt.model.model.language_model.layers[layer_source].register_forward_hook(capture_hs)

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

    patch_hook_handle = mt.model.model.language_model.layers[layer_target].register_forward_hook(patch_hs)
    
    with torch.no_grad():
        output_ids = mt.model.generate(**inp_target, max_new_tokens=max_gen_len)

    patch_hook_handle.remove()

    output = mt.processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inp_target["input_ids"], output_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    return output


