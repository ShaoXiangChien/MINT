"""Decoder-level hidden-state patching operations.

These functions implement the two-pass causal intervention protocol:
  1. **Source pass** -- run the model on a source input and cache the
     decoder hidden states at a chosen layer.
  2. **Target pass** -- run the model on a target input while replacing
     selected hidden-state positions with the cached source states,
     then generate the output.

All functions are model-agnostic: they accept an adapter and a
:class:`ModelAndTokenizer` bundle instead of touching model internals.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from src.models.base import BaseModelAdapter, ModelAndTokenizer
from .hooks import register_capture_hook, register_patch_hook, remove_hooks


def capture_decoder_hs(
    adapter: BaseModelAdapter,
    mt: ModelAndTokenizer,
    inputs: dict,
    layer: int,
) -> torch.Tensor:
    """Run a forward pass and capture the hidden states at *layer*.

    Args:
        adapter: The VLM adapter (provides ``get_decoder_layer``).
        mt: The loaded model bundle.
        inputs: Model-ready input dict.
        layer: Decoder layer index to capture from.

    Returns:
        Cloned hidden-state tensor of shape ``(batch, seq_len, hidden_dim)``.
    """
    storage: dict = {}
    target_layer = adapter.get_decoder_layer(mt, layer)
    hook = register_capture_hook(target_layer, storage)

    with torch.no_grad():
        mt.model(**adapter.get_forward_inputs(inputs))

    hook.remove()
    return storage["hidden_states"]


def patch_decoder_and_generate(
    adapter: BaseModelAdapter,
    mt: ModelAndTokenizer,
    inputs: dict,
    layer_target: int,
    cached_hs: torch.Tensor,
    patch_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_new_tokens: int = 20,
) -> str:
    """Run generation while patching decoder layer *layer_target*.

    Args:
        adapter: The VLM adapter.
        mt: The loaded model bundle.
        inputs: Model-ready input dict for the *target* prompt.
        layer_target: The decoder layer index where patching is applied.
        cached_hs: Source hidden states from :func:`capture_decoder_hs`.
        patch_fn: ``fn(target_output, cached_hs) -> patched_output``.
            Defines *which* positions to replace and how.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The decoded output string.
    """
    target_layer = adapter.get_decoder_layer(mt, layer_target)
    hook = register_patch_hook(
        target_layer, cached_hs, patch_fn, generation_mode=True,
    )

    try:
        output_text = adapter.generate(mt, inputs, max_new_tokens=max_new_tokens)
    finally:
        hook.remove()

    return output_text


# ======================================================================
# Convenience patch functions
# ======================================================================

def make_image_token_patch_fn(
    img_start: int, img_end: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a patch_fn that replaces image-token positions ``[img_start, img_end)``
    in the target with the corresponding positions from the source.

    Args:
        img_start: Inclusive start index of image tokens.
        img_end: Exclusive end index of image tokens.

    Returns:
        A callable suitable for :func:`patch_decoder_and_generate`.
    """
    def _patch(target_out: torch.Tensor, source_hs: torch.Tensor) -> torch.Tensor:
        target_out[0, img_start:img_end] = source_hs[0, img_start:img_end]
        return target_out
    return _patch


def make_full_sequence_patch_fn() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a patch_fn that replaces the entire sequence."""
    def _patch(target_out: torch.Tensor, source_hs: torch.Tensor) -> torch.Tensor:
        min_len = min(target_out.shape[1], source_hs.shape[1])
        target_out[0, :min_len] = source_hs[0, :min_len]
        return target_out
    return _patch
