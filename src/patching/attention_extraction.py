"""Attention Weight Extraction for Cross-Metric Corroboration.

This module provides utilities to extract per-layer attention weights from
VLMs during a forward pass.  The extracted weights are used to compute the
average attention that text tokens pay to image tokens at each decoder layer,
providing an independent (non-causal) signal that can be compared against the
MINT causal patching results.

If the Fusion Band hypothesis is correct, the layer range where causal
patching has the highest Override Accuracy should precisely coincide with
the layer range where text-to-image attention weight is highest.

Usage::

    from src.patching.attention_extraction import extract_text_to_image_attention

    # attn_by_layer[i] is a scalar: mean attention from text tokens to image
    # tokens at decoder layer i.
    attn_by_layer = extract_text_to_image_attention(
        adapter, mt, inputs, img_start, img_end
    )
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def extract_text_to_image_attention(
    adapter,
    mt,
    inputs: dict,
    img_start: int,
    img_end: int,
) -> List[float]:
    """Extract mean text-to-image attention weight at every decoder layer.

    Registers forward hooks on every decoder layer's self-attention module
    to capture the attention weight matrix during a single forward pass.

    Args:
        adapter: A :class:`~src.models.base.BaseModelAdapter` instance.
        mt: The loaded :class:`~src.models.base.ModelAndTokenizer` bundle.
        inputs: Model-ready input dict (from ``adapter.prepare_inputs``).
        img_start: First image token index in ``input_ids``.
        img_end: One-past-last image token index in ``input_ids``.

    Returns:
        A list of length ``num_decoder_layers``, where each element is the
        mean attention weight from all text tokens to all image tokens at
        that layer (averaged over heads and batch).

    Notes:
        - This function requires ``output_attentions=True`` to be supported
          by the model.  If the model does not support it, a fallback hook-
          based approach is used automatically.
        - The returned values are on CPU as plain Python floats.
    """
    num_layers = adapter.num_decoder_layers(mt)
    attn_by_layer: List[float] = []

    # -----------------------------------------------------------------------
    # Strategy 1: output_attentions=True (preferred, works for most HF models)
    # -----------------------------------------------------------------------
    try:
        with torch.no_grad():
            outputs = mt.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        # outputs.attentions is a tuple of length num_layers.
        # Each element has shape (batch, num_heads, seq_len, seq_len).
        if outputs.attentions is not None:
            for layer_attn in outputs.attentions:
                # layer_attn: (1, H, S, S)
                # We want attention FROM text positions TO image positions.
                seq_len = layer_attn.shape[-1]
                # Build masks for image and text positions
                image_cols = torch.zeros(seq_len, dtype=torch.bool)
                image_cols[img_start:img_end] = True
                text_rows = ~image_cols  # text tokens = non-image tokens

                # Slice: (1, H, num_text, num_image)
                text_to_image = layer_attn[0, :, text_rows, :][:, :, image_cols]
                # Mean over heads, text tokens, image tokens
                mean_val = text_to_image.mean().item()
                attn_by_layer.append(mean_val)

            return attn_by_layer

    except Exception:
        pass  # Fall through to hook-based strategy

    # -----------------------------------------------------------------------
    # Strategy 2: Hook-based extraction (fallback for models that don't
    # support output_attentions or return attentions in a non-standard way)
    # -----------------------------------------------------------------------
    captured_attns: List[torch.Tensor] = []
    hooks = []

    def _make_hook(layer_idx: int):
        def _hook(module, input, output):
            # Many attention modules return (hidden_state, attn_weights, ...)
            # or (hidden_state, None) when attn_weights are not computed.
            # We try to find a 4-D tensor in the output.
            if isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        captured_attns.append(item.detach().cpu())
                        return
            # Could not capture attention weights from this layer
            captured_attns.append(None)
        return _hook

    # Register hooks on every decoder layer's self-attention sub-module.
    # We look for a sub-module whose name contains "self_attn" or "attention".
    for i in range(num_layers):
        layer = adapter.get_decoder_layer(mt, i)
        attn_module = None
        for name, mod in layer.named_modules():
            if "self_attn" in name or name == "attention":
                attn_module = mod
                break
        if attn_module is not None:
            hooks.append(attn_module.register_forward_hook(_make_hook(i)))
        else:
            # Placeholder so indices stay aligned
            captured_attns.append(None)

    try:
        with torch.no_grad():
            mt.model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    # Process captured attention tensors
    for attn in captured_attns:
        if attn is None or attn.dim() != 4:
            attn_by_layer.append(float("nan"))
            continue
        seq_len = attn.shape[-1]
        image_cols = torch.zeros(seq_len, dtype=torch.bool)
        image_cols[img_start:img_end] = True
        text_rows = ~image_cols

        text_to_image = attn[0, :, text_rows, :][:, :, image_cols]
        attn_by_layer.append(text_to_image.mean().item())

    return attn_by_layer
