from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from general_utils import decode_tokens, make_inputs
from llava.mm_utils import tokenizer_image_token, process_images

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

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

def set_hs_patch_hooks_llava(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  """Llava patch hooks."""

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      tmp = input[0]
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = hs_
      

    def post_hook(module, input, output):

      tmp = output[0].clone()

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
        raise ValueError("Module %s not supported", module)
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
          hooks.append(model.model.layers[i].register_forward_hook(patch_hook))
        elif module == "mlp":
          hooks.append(
              model.model.layers[i].mlp.register_forward_hook(patch_hook)
          )
        elif module == "attn":
          hooks.append(
              model.model.layers[i].self_attn.register_forward_hook(patch_hook)
          )
        else:
          raise ValueError("Module %s not supported", module)
        
  return hooks

def set_hs_patch_hooks_deepseek_vl(
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
              # === 原 llama 為 model.model.layers[i]，改為 deepseek_vl 的語言模型 ===
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
              # === 原 llama 為 model.model.norm，改為 deepseek_vl 的語言模型 ===
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


# def inspect(
#     mt,
#     prompt_source,
#     prompt_target,
#     layer_source,
#     layer_target,
#     position_source,
#     position_target,
#     module="hs",
#     generation_mode=False,
#     max_gen_len=20,
#     verbose=False,
#     temperature=None,
#     source_images=None,
#     target_images=None,
# ):
#     """Inspection via patching."""
#     #########################################################
#     # Make inputs for source and target prompts
#     #########################################################
#     if target_images is None:
#         input_ids_target = (
#             tokenizer_image_token(prompt_target, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .to(mt.device)
#         )
#         inp_target = {
#             "input_ids": input_ids_target,
#             "images": None,
#             "image_sizes": None
#         }
#     else:
#         images_tensor_target = process_images(target_images, mt.image_processor, mt.model.config).to(mt.device, dtype=torch.float16)
#         image_sizes_target = [image.size for image in target_images]
#         input_ids_target = (
#             tokenizer_image_token(prompt_target, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .to(mt.device)
#         )
#         inp_target = {
#             "input_ids": input_ids_target,
#             "images": images_tensor_target,
#             "image_sizes": image_sizes_target
#         }

    


#     if source_images is None:
#         input_ids_source = (
#             tokenizer_image_token(prompt_source, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .to(mt.device)
#         )
#         inp_source = {
#             "input_ids": input_ids_source,
#             "images": None,
#             "image_sizes": None
#         }

#     else:
#         images_tensor_source = process_images(source_images, mt.image_processor, mt.model.config).to(mt.device, dtype=torch.float16)
#         image_sizes_source = [image.size for image in source_images]
#         input_ids_source = (
#             tokenizer_image_token(prompt_source, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#             .unsqueeze(0)
#             .to(mt.device)
#         )
#         inp_source = {
#            "input_ids": input_ids_source,
#            "images": images_tensor_source,
#            "image_sizes": image_sizes_source
#            }
    
#     if position_target < 0:
#         position_target = len(inp_target["input_ids"][0]) + position_target

#     if position_source < 0:
#         position_source = len(inp_source["input_ids"][0]) + position_source
        
#     def custom_decode(tokenizer, token_id):
#         if token_id == -200:  # Replace with IMAGE_TOKEN_INDEX if it's defined elsewhere
#             return "<image>"
#         else:
#             return tokenizer.decode([token_id], skip_special_tokens=False)
        

#     if verbose:
#         if source_images is None:
#             print("prompt_patch:", [custom_decode(mt.tokenizer, x) for x in inp_source["input_ids"][0]])
#         else:
#             print("prompt_patch (multimodal):", [custom_decode(mt.tokenizer, x) for x in inp_source["input_ids"][0]])
#             print(mt.tokenizer.convert_ids_to_tokens(inp_source["input_ids"][0]))
    
#     #########################################################
#     # First run the model on the source prompt
#     #########################################################
#     hs_cache_ = []
#     store_hooks = []
#     if module == "mlp":
#         def store_mlp_hook(module, input, output):
#             hs_cache_.append(output[0])

#         for layer in mt.model.model.layers:
#             store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))
#     elif module == "attn":
#         def store_attn_hook(module, input, output):
#             hs_cache_.append(output[0].squeeze())

#         for layer in mt.model.model.layers:
#             store_hooks.append(layer.self_attn.register_forward_hook(store_attn_hook))
    
#     output = mt.model(**inp_source, output_hidden_states=True)

#     if module == "hs":
#         hs_cache_ = [output["hidden_states"][layer + 1][0] for layer in range(mt.num_layers)]

#     remove_hooks(store_hooks)

#     #########################################################
#     # Set up patching configuration
#     #########################################################
#     L_source = len(inp_source["input_ids"][0])
#     sequence_length_source = len(hs_cache_[0])  # From source model output
#     num_patches = sequence_length_source - L_source + 1

#     img_pos_source = (inp_source["input_ids"][0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
#     img_pos_target = (inp_target["input_ids"][0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
    
#     hs_patch_config = {
#     layer_target: [
#         (img_pos_target + i, hs_cache_[layer_source][img_pos_source + i])
#         for i in range(num_patches)
#     ]
#     }


#     # Apply patches
#     patch_hooks = mt.set_hs_patch_hooks(
#         mt.model,
#         hs_patch_config,
#         module=module,
#         patch_input=False,
#         skip_final_ln=(layer_source == layer_target == mt.num_layers - 1),
#         generation_mode=True
#     )

#     #########################################################
#     # Generate output with patching
#     #########################################################
#     if generation_mode:
#         output_ids = mt.model.generate(
#             inp_target["input_ids"],
#             images=inp_target["images"],
#             image_sizes=inp_target["image_sizes"],
#             do_sample=True if temperature else False,
#             temperature=temperature if temperature else 0.7,
#             max_new_tokens=max_gen_len,
#             pad_token_id=mt.tokenizer.pad_token_id,
#         )
#         output = mt.tokenizer.decode(output_ids[0][1:])
#     else:
#         output = mt.model(**inp_target)
#         answer_prob, answer_t = torch.max(torch.softmax(output.logits[0, -1, :], dim=0), dim=0)
#         output = (decode_tokens(mt.tokenizer, [answer_t])[0], round(answer_prob.cpu().item(), 4))

#     if verbose:
#         print("prediction with patching:", output)
        
#     # Clean up hooks
#     remove_hooks(patch_hooks)
    
#     return output

def inspect(
    mt,
    prompt_source,
    prompt_target,
    vision_layer_source,  # Layer in source vision tower to extract from (0 to 23)
    generation_mode=False,
    max_gen_len=20,
    verbose=False,
    temperature=None,
    source_images=None,
    target_images=None,
    adv_tensor=None,
):
    """Inspection by patching source vision tower layer output into target before projection."""
    #########################################################
    # Prepare inputs for source and target prompts
    #########################################################
    if target_images is None:
        input_ids_target = (
            tokenizer_image_token(prompt_target, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(mt.device)
        )
        inp_target = {
            "input_ids": input_ids_target,
            "images": None,
            "image_sizes": None
        }
    else:
        images_tensor_target = process_images(target_images, mt.image_processor, mt.model.config).to(mt.device, dtype=torch.float16)
        image_sizes_target = [image.size for image in target_images]
        input_ids_target = (
            tokenizer_image_token(prompt_target, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(mt.device)
        )
        inp_target = {
            "input_ids": input_ids_target,
            "images": images_tensor_target,
            "image_sizes": image_sizes_target
        }

    if source_images is None:
        input_ids_source = (
            tokenizer_image_token(prompt_source, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(mt.device)
        )
        inp_source = {
            "input_ids": input_ids_source,
            "images": None,
            "image_sizes": None
        }
    else:
        if adv_tensor is not None:
            images_tensor_source = adv_tensor
        else:
            images_tensor_source = process_images(source_images, mt.image_processor, mt.model.config).to(mt.device, dtype=torch.float16)
        
        image_sizes_source = [image.size for image in source_images]
        input_ids_source = (
            tokenizer_image_token(prompt_source, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(mt.device)
        )
        inp_source = {
            "input_ids": input_ids_source,
            "images": images_tensor_source,
            "image_sizes": image_sizes_source
        }

    #########################################################
    # Capture hidden states from specified layer in source vision tower
    #########################################################
    source_vision_hidden = None

    def capture_vision_layer(module, input, output):
        nonlocal source_vision_hidden
        source_vision_hidden = output

    # Register hook on the specified layer in the source vision tower
    capture_hook = mt.model.model.vision_tower.vision_tower.vision_model.encoder.layers[vision_layer_source].register_forward_hook(capture_vision_layer)

    # Run forward pass on source input to capture the hidden states
    with torch.no_grad():
        _ = mt.model(**inp_source)

    capture_hook.remove()

    # source_vision_hidden is now the hidden states from vision_layer_source (shape: [1, 576, 1024])

    #########################################################
    # Patch the target vision tower output before projection
    #########################################################
    def replace_vision_output(module, input, output):
        # Replace the entire vision tower output with the source layer's hidden states
        # Convert tuple to tensor if needed
        if isinstance(source_vision_hidden, tuple):
            return source_vision_hidden[0]  # Take first element if tuple
        return source_vision_hidden

    # Register hook on the vision tower’s forward method (after all layers, before mm_projector)
    replace_hook = mt.model.model.vision_tower.register_forward_hook(replace_vision_output)

    #########################################################
    # Generate output with patched vision tower output
    #########################################################
    if generation_mode:
        output_ids = mt.model.generate(
            inp_target["input_ids"],
            images=inp_target["images"],
            image_sizes=inp_target["image_sizes"],
            do_sample=True if temperature else False,
            temperature=temperature if temperature else 0.7,
            max_new_tokens=max_gen_len,
            pad_token_id=mt.tokenizer.pad_token_id,
        )
        output = mt.tokenizer.decode(output_ids[0][1:])
    else:
        output = mt.model(**inp_target)
        answer_prob, answer_t = torch.max(torch.softmax(output.logits[0, -1, :], dim=0), dim=0)
        output = (decode_tokens(mt.tokenizer, [answer_t])[0], round(answer_prob.cpu().item(), 4))

    if verbose:
        print(f"Prediction with vision tower patched from source layer {vision_layer_source}:", output)

    # Clean up hook
    replace_hook.remove()

    return output


def inspect_vision_in_lm(
    mt,
    prompt_source,
    prompt_target,
    source_images,
    target_images,
    layer_source,
    layer_target,
    max_gen_len=20,
    verbose=False,
    temperature=None,
):
    """
    Inspect the vision features in the language model.
    """

    # prepare inputs
    image_tensor_target = process_images(target_images, mt.image_processor, mt.model.config).to(mt.device, dtype=torch.float16)
    image_tensor_target = image_tensor_target.to(mt.device, dtype=torch.float16)
    image_sizes_target = [image.size for image in target_images]

    image_tensor_source = process_images(source_images, mt.image_processor, mt.model.config).to(mt.device, dtype=torch.float16)
    image_tensor_source = image_tensor_source.to(mt.device, dtype=torch.float16)
    image_sizes_source = [image.size for image in source_images]

    input_ids_target = (
        tokenizer_image_token(prompt_target, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(mt.device)
    )

    inp_target = {
        "input_ids": input_ids_target,
        "images": image_tensor_target,
        "image_sizes": image_sizes_target
    }

    input_ids_source = (
        tokenizer_image_token(prompt_source, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(mt.device)
    )

    inp_source = {
        "input_ids": input_ids_source,
        "images": image_tensor_source,
        "image_sizes": image_sizes_source
    }

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
        nonlocal inp_source
        
        if patched:
            return output
        
        img_start_idx = torch.where(inp_source["input_ids"][0] == IMAGE_TOKEN_INDEX)[0][0]
        seq_after_img = len(inp_source["input_ids"][0]) - (img_start_idx + 1)
        img_end_idx = extracted_activations["image_features"][0].shape[0] - seq_after_img - 1

        extracted_features = extracted_activations["image_features"][0].to(mt.device)
    
        for i in range(img_start_idx, img_end_idx):
            output[0][0][i] = extracted_features[i]


        patched = True
      

        return output

    patch_hook_handle = mt.model.model.layers[layer_target].register_forward_hook(patch_hs)
    
    with torch.no_grad():
        output_ids = mt.model.generate(
            inp_target["input_ids"],
            images=inp_target["images"],
            image_sizes=inp_target["image_sizes"],
            do_sample=True if temperature else False,
            temperature=temperature if temperature else 0.7,
            max_new_tokens=max_gen_len,
            pad_token_id=mt.tokenizer.pad_token_id,
        )

    patch_hook_handle.remove()

    output = mt.tokenizer.decode(output_ids[0][1:])

    return output



# Adding support for batched patching. More than 10x speedup
def set_hs_patch_hooks_llava_batch(
    model,
    hs_patch_config,
    module="hs",       # Only "hs" supported here for clarity
    patch_input=False,
    generation_mode=False,
):
    """
    LLaVA patch hooks with batch support.

    Args:
        model: Your LLaVA model.
        hs_patch_config (List[dict]): A list of dicts, where each dict typically has:
            {
                "batch_idx": (int) which batch item to patch
                "layer_target": (int) which layer to patch
                "position_target": (int) which sequence position to patch
                "hidden_rep": (torch.Tensor) the hidden representation to inject
                "skip_final_ln": (bool) whether to skip the final layer norm patch
            }
        module (str): Currently "hs" is supported in this example, but you could
            adapt it for "mlp"/"attn" if needed, mirroring the structure above.
        patch_input (bool): Whether to patch the layer input (pre_hook) or the layer
            output (post_hook).
        generation_mode (bool): If True, we skip patching steps with seq_len == 1,
            which typically happens after the first pass of `model.generate()`.

    Returns:
        hooks (List[torch.utils.hooks.RemovableHandle]): A list of hook handles. Make
        sure to remove them when you are done (e.g. via `remove_hooks(hooks)`).
    """

    if module != "hs":
        raise ValueError(f"Module {module} not yet supported for batch patching.")

    def patch_hs(name, position_hs_dict, patch_input, generation_mode):
        """
        Returns a hook (pre- or post-forward) that patches a single item in the batch.
        """
        def pre_hook(layer_module, inputs):
            """
            Pre-hook: patch the input hidden state before the layer forward pass.
            `inputs` is typically a tuple; we care about inputs[0] which is
            (batch_size, seq_len, hidden_dim).
            """
            idx_ = position_hs_dict["batch_idx"]
            position_ = position_hs_dict["position_target"]
            hs_ = position_hs_dict["hidden_rep"]

            # If generation_mode is True and the model is generating step-by-step,
            # after the first pass the seq_len might be 1 for subsequent tokens.
            # We skip patching if seq_len == 1.
            input_len = len(inputs[0][idx_])
            if generation_mode and input_len == 1:
                return  # do nothing
            inputs[0][idx_][position_] = hs_

        def post_hook(layer_module, inputs, output):
            """
            Post-hook: patch the output hidden state after the layer forward pass.
            """
            idx_ = position_hs_dict["batch_idx"]
            position_ = position_hs_dict["position_target"]
            hs_ = position_hs_dict["hidden_rep"]

            # Depending on the layer type, output could be just a Tensor of shape
            # (batch, seq_len, hidden_dim), or a tuple. In your code, if "skip_ln"
            # or "mlp" is in the name, the shape might differ. For standard "hs":
            # output is typically (Tensor of shape (batch, seq_len, hidden_dim)) or
            # a tuple whose first element is the hidden states. Here we assume the
            # LLaVA blocks follow the same pattern as standard LLaMA/HF.
            if "skip_ln" in name or "mlp" in name:
                # output: (batch, seq_len, hidden_dim)
                output_len = len(output[idx_])
                if generation_mode and output_len == 1:
                    return
                output[idx_][position_] = hs_
            else:
                # output[0]: (batch, seq_len, hidden_dim)
                output_len = len(output[0][idx_])
                if generation_mode and output_len == 1:
                    return
                output[0][idx_][position_] = hs_

        return pre_hook if patch_input else post_hook

    hooks = []
    for item in hs_patch_config:
        i = item["layer_target"]
        skip_final_ln = item["skip_final_ln"]
        hook_fn = patch_hs(f"patch_{module}_{i}", item, patch_input, generation_mode)

        if patch_input:
            # Patching the input hidden states (pre-hook)
            hooks.append(
                model.model.layers[i].register_forward_pre_hook(hook_fn)
            )
        else:
            # Patching the output hidden states (post-hook)
            # If skip_final_ln is True and i == last layer, patch the final LN instead
            if skip_final_ln and i == len(model.model.layers) - 1:
                hooks.append(
                    model.model.norm.register_forward_hook(
                        patch_hs(
                            f"patch_{module}_{i}_skip_ln", 
                            item, 
                            patch_input, 
                            generation_mode
                        )
                    )
                )
            else:
                hooks.append(
                    model.model.layers[i].register_forward_hook(hook_fn)
                )

    return hooks

def set_hs_patch_hooks_deepseek_vl_batch(
    model,
    hs_patch_config,
    module="hs",       # "hs", "mlp", or "attn"
    patch_input=False,
    generation_mode=False,
):
    """
    DeepSeek-VL patch hooks with batch support.

    This function supports injecting hidden states for multiple samples
    in a batch in one forward pass.

    Args:
        model: Your DeepSeek-VL model instance. We assume that the
            language model lives in model.language_model.model (similar
            to the code you provided).
        hs_patch_config (List[dict]):
            A list of dictionaries, each describing how to patch exactly
            one sample in the batch. For example:
                {
                    "batch_idx": (int) which sample in the batch to patch,
                    "layer_target": (int) which layer to patch,
                    "position_target": (int) which sequence position to patch,
                    "hidden_rep": (torch.Tensor) the hidden representation to inject,
                    "skip_final_ln": (bool) whether to skip final layer norm,
                }
        module (str):
            Which part to patch:
                - "hs": default hidden-state patch (patch the outputs of the entire layer)
                - "mlp": patch MLP output only
                - "attn": patch self-attention output only
        patch_input (bool):
            If True, apply the patch to the layer's input (pre-hook).
            Otherwise, apply the patch after the layer's forward pass (post-hook).
        generation_mode (bool):
            If True, skip patching if the sequence length is 1 (commonly happens
            during step-by-step generation after the first pass).

    Returns:
        hooks (List[torch.utils.hooks.RemovableHandle]): A list of hook handles.
        Make sure to remove them (via `hook.remove()`) when you're done patching.
    """

    def patch_hs(name, position_hs, patch_input, generation_mode):
        """
        Creates a hook function (either pre- or post-forward) to patch a single
        batch index.
        """
        def pre_hook(layer_module, inputs):
            """
            Pre-hook: Patch the input hidden states before the layer forward pass.
            `inputs[0]` typically has shape (batch_size, seq_len, hidden_dim).
            """
            idx_ = position_hs["batch_idx"]
            position_ = position_hs["position_target"]
            hs_ = position_hs["hidden_rep"]

            # Check for generation mode skip
            # If generation_mode = True and seq_len == 1, do nothing
            seq_len = len(inputs[0][idx_])
            if generation_mode and seq_len == 1:
                return
            # Patch the hidden state
            inputs[0][idx_][position_] = hs_

        def post_hook(layer_module, inputs, output):
            """
            Post-hook: Patch the output hidden states after the layer forward pass.
            Depending on the layer, `output` could be:
              - a single Tensor of shape (batch_size, seq_len, hidden_dim), or
              - a tuple where output[0] is the Tensor.
            """
            idx_ = position_hs["batch_idx"]
            position_ = position_hs["position_target"]
            hs_ = position_hs["hidden_rep"]

            # If you detect "skip_ln" or "mlp" in the name,
            # you might need to patch output directly.
            if "skip_ln" in name or "mlp" in name:
                # output: (batch, seq_len, hidden_dim)
                seq_len = len(output[idx_])
                if generation_mode and seq_len == 1:
                    return
                output[idx_][position_] = hs_
            else:
                # Otherwise assume the first element in output is the hidden states
                # output[0]: (batch, seq_len, hidden_dim)
                seq_len = len(output[0][idx_])
                if generation_mode and seq_len == 1:
                    return
                output[0][idx_][position_] = hs_

        return pre_hook if patch_input else post_hook

    hooks = []
    # Loop over each patch specification in hs_patch_config
    for item in hs_patch_config:
        i = item["layer_target"]
        skip_final_ln = item["skip_final_ln"]
        # Construct the hook function for this item
        hook_fn = patch_hs(f"patch_{module}_{i}", item, patch_input, generation_mode)

        # Decide where to attach the hook
        if patch_input:
            # Register a pre-hook at the designated submodule
            if module == "hs":
                hooks.append(
                    model.language_model.model.layers[i].register_forward_pre_hook(hook_fn)
                )
            elif module == "mlp":
                hooks.append(
                    model.language_model.model.layers[i].mlp.register_forward_pre_hook(hook_fn)
                )
            elif module == "attn":
                hooks.append(
                    model.language_model.model.layers[i].self_attn.register_forward_pre_hook(hook_fn)
                )
            else:
                raise ValueError(f"Module {module} not supported in pre-hook.")
        else:
            # Patching output after the layer forward pass
            # If skip_final_ln is True and i is the last layer, patch final LN instead
            if skip_final_ln and i == len(model.language_model.model.layers) - 1 and module == "hs":
                hooks.append(
                    model.language_model.model.norm.register_forward_hook(
                        patch_hs(f"patch_hs_{i}_skip_ln", item, patch_input, generation_mode)
                    )
                )
            else:
                if module == "hs":
                    hooks.append(
                        model.language_model.model.layers[i].register_forward_hook(hook_fn)
                    )
                elif module == "mlp":
                    hooks.append(
                        model.language_model.model.layers[i].mlp.register_forward_hook(hook_fn)
                    )
                elif module == "attn":
                    hooks.append(
                        model.language_model.model.layers[i].self_attn.register_forward_hook(hook_fn)
                    )
                else:
                    raise ValueError(f"Module {module} not supported in post-hook.")

    return hooks


def inspect_batch(mt, df, batch_size=256, transform=None, module="hs"):
  """Inspects batch: source/target layer/position could differ within batch."""
  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  generations = []

  def _inspect_single_batch(batch_df):
    batch_size = len(batch_df)
    prompt_source_batch = np.array(batch_df["prompt_source"])
    prompt_target_batch = np.array(batch_df["prompt_target"])
    layer_source_batch = np.array(batch_df["layer_source"])
    layer_target_batch = np.array(batch_df["layer_target"])
    position_source_batch = np.array(batch_df["position_source"])
    position_target_batch = np.array(batch_df["position_target"])
    max_gen_len = np.array(batch_df["max_gen_len"])

    # adjust position_target to be absolute rather than relative
    inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
    for i in range(batch_size):
      if position_target_batch[i] < 0:
        position_target_batch[i] += len(inp_target["input_ids"][i])

    # first run the the model on without patching and get the results.
    inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
    output_orig = mt.model(**inp_source, output_hidden_states=True)

    # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
    hidden_rep = [
        output_orig.hidden_states[layer_source_batch[i] + 1][i][
            position_source_batch[i]
        ]
        for i in range(batch_size)
    ]
    if transform is not None:
      for i in range(batch_size):
        hidden_rep[i] = transform(hidden_rep[i])

    # now do a second run on prompt, while patching the input hidden state.
    hs_patch_config = [
        {
            "batch_idx": i,
            "layer_target": layer_target_batch[i],
            "position_target": position_target_batch[i],
            "hidden_rep": hidden_rep[i],
            "skip_final_ln": (
                layer_source_batch[i]
                == layer_target_batch[i]
                == mt.num_layers - 1
            ),
        }
        for i in range(batch_size)
    ]
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        generation_mode=True,
    )

    # NOTE: inputs are left padded,
    # and sequence length is the same across batch
    # to support generations of variable lengths,
    # first generate with maximum number of tokens needed in the batch
    seq_len = len(inp_target["input_ids"][0])
    output_toks = mt.model.generate(
        inp_target["input_ids"],
        max_length=seq_len + max(max_gen_len),
        pad_token_id=mt.model.generation_config.eos_token_id,
    )[:, seq_len:]

    # then, we select only the subset of tokens that we need
    generations = [
        mt.tokenizer.decode(output_toks[i][: max_gen_len[i]])
        for i in range(batch_size)
    ]

    # remove patching hooks
    remove_hooks(patch_hooks)

    return generations

  for i in tqdm.tqdm(range(1 + len(df) // batch_size)):
    cur_df = df.iloc[batch_size * i : batch_size * (i + 1)]
    batch_generations = _inspect_single_batch(cur_df)
    generations.extend(batch_generations)

  return generations


def evaluate_attriburte_exraction_batch(
    mt,
    df,
    batch_size=256,
    max_gen_len=10,
    transform=None,
    is_icl=True,
    module="hs",
):
  """Evaluates attribute extraction with batch support."""
  # We don't know the exact token position of the
  # attribute, as it is not necessarily the next token. So, precision and
  # surprisal may not apply directly.

  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  def _evaluate_attriburte_exraction_single_batch(batch_df):
    batch_size = len(batch_df)
    prompt_source_batch = np.array(batch_df["prompt_source"])
    prompt_target_batch = np.array(batch_df["prompt_target"])
    layer_source_batch = np.array(batch_df["layer_source"])
    layer_target_batch = np.array(batch_df["layer_target"])
    position_source_batch = np.array(batch_df["position_source"])
    position_target_batch = np.array(batch_df["position_target"])

    object_batch = np.array(batch_df["object"])

    # Adjust position_target to be absolute rather than relative
    inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
    for i in range(batch_size):
      if position_target_batch[i] < 0:
        position_target_batch[i] += len(inp_target["input_ids"][i])

    # Step 1: run model on source prompt without patching and get the hidden
    # representations.
    inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
    output_orig = mt.model(**inp_source, output_hidden_states=True)

    # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
    #         hidden_rep = []
    #         for i in range(batch_size):
    #             hidden_rep.append(output_orig.hidden_states[layer_source_batch[i] + 1][i][position_source_batch[i]])
    hidden_rep = []
    for i in range(batch_size):
        layer_idx = min(layer_source_batch[i] + 1, len(output_orig.hidden_states) - 1)
        seq_len = output_orig.hidden_states[layer_idx][i].size(0)
        pos_idx = min(position_source_batch[i], seq_len - 1)
        hidden_rep.append(output_orig.hidden_states[layer_idx][i][pos_idx])
        
    if transform is not None:
      for i in range(batch_size):
        hidden_rep[i] = transform(hidden_rep[i])

    # Step 2: Do second run on target prompt, while patching the input
    # hidden state.
    hs_patch_config = [
        {
            "batch_idx": i,
            "layer_target": layer_target_batch[i],
            "position_target": position_target_batch[i],
            "hidden_rep": hidden_rep[i],
            "skip_final_ln": (
                layer_source_batch[i]
                == layer_target_batch[i]
                == mt.num_layers - 1
            ),
        }
        for i in range(batch_size)
    ]
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        generation_mode=True,
    )

    # Note that inputs are left padded,
    # and sequence length is the same across batch
    seq_len = len(inp_target["input_ids"][0])
    output_toks = mt.model.generate(
        inp_target["input_ids"],
        max_length=seq_len + max_gen_len,
        pad_token_id=mt.model.generation_config.eos_token_id,
    )[:, seq_len:]
    generations_patched = decode_tokens(mt.tokenizer, output_toks)
    if is_icl:
      prefix = batch_df["prefix"].iloc[0]

      def _crop_by_prefix(generations, prefix):
        concatenated_str = " ".join(generations)
        _pos = concatenated_str.find(prefix)
        return concatenated_str[:_pos]

      generations_patched_postprocessed = np.array([
          _crop_by_prefix(generations_patched[i], prefix)
          for i in range(batch_size)
      ])
    else:
      generations_patched_postprocessed = np.array(
          [" ".join(generations_patched[i]) for i in range(batch_size)])

    is_correct_patched = np.array([
        object_batch[i].replace(" ", "")
        in generations_patched_postprocessed[i].replace(" ", "")
        for i in range(batch_size)
    ])

    # remove patching hooks
    remove_hooks(patch_hooks)

    cpu_hidden_rep = np.array(
        [hidden_rep[i].detach().cpu().numpy() for i in range(batch_size)]
    )

    results = {
        "generations_patched": generations_patched,
        "generations_patched_postprocessed": generations_patched_postprocessed,
        "is_correct_patched": is_correct_patched,
        "hidden_rep": cpu_hidden_rep,
    }

    return results

  results = {}
  n_batches = len(df) // batch_size
  if len(df) % batch_size != 0:
    n_batches += 1
  for i in tqdm(range(len(df) // batch_size)):
    cur_df = df.iloc[batch_size * i : batch_size * (i + 1)]
    batch_results = _evaluate_attriburte_exraction_single_batch(cur_df)
    for key, value in batch_results.items():
      if key in results:
        results[key] = np.concatenate((results[key], value))
      else:
        results[key] = value

  return results