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

"""Unified forward-hook utilities for hidden-state patching.

These hooks are model-agnostic: callers pass the target ``nn.Module``
(obtained via a model adapter) rather than hard-coding architecture paths.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def register_capture_hook(
    layer: nn.Module,
    storage: dict,
    key: str = "hidden_states",
) -> torch.utils.hooks.RemovableHook:
    """Attach a forward hook that stores the layer output in *storage*.

    Args:
        layer: The ``nn.Module`` to hook (e.g. a decoder layer).
        storage: A mutable dict; upon forward, ``storage[key]`` will be
            set to the cloned output tensor.
        key: The dict key to write into.

    Returns:
        A removable hook handle.
    """
    def _hook(module, input, output):
        if key not in storage:
            out_tensor = output[0] if isinstance(output, tuple) else output
            storage[key] = out_tensor.clone()

    return layer.register_forward_hook(_hook)


def register_patch_hook(
    layer: nn.Module,
    cached_hs: torch.Tensor,
    patch_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    generation_mode: bool = True,
) -> torch.utils.hooks.RemovableHook:
    """Attach a forward hook that patches the layer output using *patch_fn*.

    During autoregressive generation the model calls each layer multiple
    times with ``seq_len == 1`` (the KV-cache path).  When
    ``generation_mode=True`` those single-token calls are skipped so that
    patching only happens on the initial full-sequence forward pass.

    Args:
        layer: The ``nn.Module`` to hook.
        cached_hs: The source hidden states to patch from
            (shape ``(batch, seq, hidden)`` or ``(seq, hidden)``).
        patch_fn: ``fn(target_output, cached_hs) -> patched_output``.
            Called once with the live output tensor and the cached source.
        generation_mode: If True, skip single-token (KV-cache) forward calls.

    Returns:
        A removable hook handle.
    """
    patched = False

    def _hook(module, input, output):
        nonlocal patched
        if patched:
            return output

        out_tensor = output[0] if isinstance(output, tuple) else output

        # Skip KV-cache-only steps during generation
        if generation_mode and out_tensor.shape[-2] == 1:
            return output

        patched_tensor = patch_fn(out_tensor, cached_hs)

        if isinstance(output, tuple):
            return (patched_tensor,) + output[1:]

        patched = True
        return patched_tensor

    return layer.register_forward_hook(_hook)


def register_position_patch_hooks(
    layer: nn.Module,
    position_hs_pairs: List[Tuple[int, torch.Tensor]],
    generation_mode: bool = True,
) -> torch.utils.hooks.RemovableHook:
    """Patch specific token positions with pre-computed hidden states.

    This replicates the ``set_hs_patch_hooks_*`` pattern from the original
    per-model patchscopes utilities.

    Args:
        layer: The decoder layer ``nn.Module`` to hook.
        position_hs_pairs: List of ``(position_idx, hidden_state_tensor)``
            tuples.  Each position in the layer output will be overwritten
            with the corresponding cached vector.
        generation_mode: If True, skip single-token (KV-cache) forward calls.

    Returns:
        A removable hook handle.
    """
    def _hook(module, input, output):
        out_tensor = output[0] if isinstance(output, tuple) else output

        if generation_mode and out_tensor.shape[-2] == 1:
            return output

        for pos, hs in position_hs_pairs:
            out_tensor[0, pos] = hs

        if isinstance(output, tuple):
            return (out_tensor,) + output[1:]
        return out_tensor

    return layer.register_forward_hook(_hook)


def remove_hooks(hooks: List[torch.utils.hooks.RemovableHook]) -> None:
    """Remove a list of registered hooks."""
    for h in hooks:
        h.remove()
