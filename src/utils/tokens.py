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

"""Shared token manipulation utilities.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import math
import torch


def make_inputs(tokenizer, prompts, device="cuda"):
    """Tokenize a list of text prompts with left-padding and attention masks.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        prompts: List of prompt strings.
        device: Target device for output tensors.

    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors,
        each of shape ``(batch, max_seq_len)``.
    """
    for i, prompt in enumerate(prompts):
        if prompt is None or (isinstance(prompt, float) and math.isnan(prompt)):
            raise ValueError(f"Invalid prompt at index {i}: cannot be NaN or None")
        if not isinstance(prompt, str):
            prompts[i] = str(prompt)

    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)

    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0

    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]

    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    """Decode a token id array into a list of individual token strings.

    Recursively handles batched (2-D) inputs.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        token_array: 1-D or 2-D tensor / list of token ids.

    Returns:
        List of decoded token strings (1-D input) or list-of-lists (2-D).
    """
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    """Find the token span corresponding to *substring* within *token_array*.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        token_array: 1-D tensor of token ids.
        substring: The text substring to locate.

    Returns:
        Tuple ``(tok_start, tok_end)`` giving the half-open token range.
    """
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
    """Run a forward pass and return the greedy next-token prediction.

    Args:
        model: HuggingFace causal language model.
        inp: Dict with at least 'input_ids' tensor.

    Returns:
        Tuple ``(predicted_ids, probabilities)`` each of shape ``(batch,)``.
    """
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def set_requires_grad(requires_grad, *models):
    """Toggle ``requires_grad`` for all parameters in the given models."""
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            raise TypeError(f"Unknown type {type(model)!r}")
