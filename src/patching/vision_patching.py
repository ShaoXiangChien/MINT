"""Vision-encoder-level hidden-state patching operations.

These functions operate on the vision tower (e.g. ViT blocks) rather than
the language decoder.  The two-pass protocol is the same:
  1. Capture vision embeddings from a source image at a given ViT layer.
  2. Inject those embeddings into a target forward pass at a (possibly
     different) ViT layer, then generate.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.models.base import BaseModelAdapter, ModelAndTokenizer
from .hooks import register_capture_hook, remove_hooks


def capture_vision_emb(
    adapter: BaseModelAdapter,
    mt: ModelAndTokenizer,
    inputs: dict,
    vision_layer: int,
) -> torch.Tensor:
    """Run a forward pass and capture the output of vision layer *vision_layer*.

    Args:
        adapter: The VLM adapter.
        mt: The loaded model bundle.
        inputs: Model-ready input dict (must contain image data).
        vision_layer: Vision encoder layer index.

    Returns:
        Cloned embedding tensor from the specified vision layer.
    """
    storage: dict = {}
    target_layer = adapter.get_vision_layer(mt, vision_layer)
    hook = register_capture_hook(target_layer, storage)

    with torch.no_grad():
        mt.model(**inputs)

    hook.remove()
    return storage["hidden_states"]


def patch_vision_and_generate(
    adapter: BaseModelAdapter,
    mt: ModelAndTokenizer,
    inputs: dict,
    vision_layer_target: int,
    cached_emb: torch.Tensor,
    patch_indices: Optional[List[int]] = None,
    max_new_tokens: int = 20,
) -> str:
    """Generate while patching a vision encoder layer with cached embeddings.

    If *patch_indices* is ``None``, the **entire** layer output is replaced.
    Otherwise only the specified patch-token indices are overwritten (useful
    for object-level patching with bounding-box-derived indices).

    Args:
        adapter: The VLM adapter.
        mt: The loaded model bundle.
        inputs: Model-ready input dict for the *target* image.
        vision_layer_target: Vision layer to patch.
        cached_emb: Source embeddings from :func:`capture_vision_emb`.
        patch_indices: Optional list of patch-token indices to replace.
            If ``None``, the full output is replaced.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The decoded output string.
    """
    patched = False

    def _hook(module, input, output):
        nonlocal patched
        if patched:
            return output

        out_tensor = output[0] if isinstance(output, tuple) else output

        if patch_indices is not None:
            for idx in patch_indices:
                out_tensor[idx - 1] = cached_emb[idx - 1]
        else:
            if isinstance(output, tuple):
                return (cached_emb,) + output[1:]
            return cached_emb

        patched = True
        if isinstance(output, tuple):
            return (out_tensor,) + output[1:]
        return out_tensor

    target_layer = adapter.get_vision_layer(mt, vision_layer_target)
    hook = target_layer.register_forward_hook(_hook)

    try:
        output_text = adapter.generate(mt, inputs, max_new_tokens=max_new_tokens)
    finally:
        hook.remove()

    return output_text
