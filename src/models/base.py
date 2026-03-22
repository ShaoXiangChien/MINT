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

"""Base classes for model adapters.

Every VLM family (LLaVA, DeepSeek-VL2, Qwen2-VL, InternVL) exposes the
same interface through a concrete subclass of :class:`BaseModelAdapter`.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
from PIL import Image

from src.utils.tokens import set_requires_grad


class ModelAndTokenizer:
    """Lightweight container that bundles a VLM, its tokenizer/processor,
    and metadata needed by the patching engine.

    Attributes:
        model: The loaded ``nn.Module``.
        tokenizer: HuggingFace tokenizer (may be ``None`` for processor-only models).
        processor: HuggingFace processor (may be ``None`` for tokenizer-only models).
        image_processor: LLaVA-style image processor (may be ``None``).
        device: The primary device string (e.g. ``"cuda:0"``).
        num_layers: Number of decoder layers detected.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[nn.Module] = None,
        tokenizer=None,
        processor=None,
        image_processor=None,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = image_processor
        self.device = device

        self.layer_names = _detect_layer_names(model)
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model={type(self.model).__name__}, "
            f"layers={self.num_layers}, device={self.device})"
        )


def _detect_layer_names(model: nn.Module) -> list[str]:
    """Auto-detect decoder layer names for both direct and wrapped architectures."""
    # Models like DeepSeek-VL2 and InternVL wrap the LM inside model.language_model
    if hasattr(model, "language_model"):
        names = [
            n for n, _ in model.language_model.named_modules()
            if re.match(r"^(model)\.(layers)\.\d+$", n)
        ]
        if names:
            return names

    return [
        n for n, _ in model.named_modules()
        if re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n)
    ]


class BaseModelAdapter(ABC):
    """Abstract interface that every VLM adapter must implement.

    The adapter encapsulates model-specific details (loading, input
    preparation, architecture paths) so that experiment scripts and the
    patching engine can remain model-agnostic.
    """

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @abstractmethod
    def load_model(self, model_path: str, device: str) -> ModelAndTokenizer:
        """Load the VLM and return a :class:`ModelAndTokenizer` bundle."""
        ...

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    @abstractmethod
    def prepare_inputs(self, prompt: str, image: Image.Image,
                       mt: ModelAndTokenizer) -> dict:
        """Build the model-ready input dict from a text prompt and an image.

        Args:
            prompt: The text prompt.
            image: A PIL Image (already in RGB).
            mt: The loaded model bundle (provides processor / tokenizer refs).

        Returns:
            A dict that can be unpacked into ``model(**inputs)`` or
            ``model.generate(**inputs)``.
        """
        ...

    # ------------------------------------------------------------------
    # Architecture accessors
    # ------------------------------------------------------------------
    @abstractmethod
    def get_decoder_layer(self, mt: ModelAndTokenizer, layer_idx: int) -> nn.Module:
        """Return the ``nn.Module`` for decoder layer *layer_idx*."""
        ...

    @abstractmethod
    def get_vision_layer(self, mt: ModelAndTokenizer, layer_idx: int) -> nn.Module:
        """Return the ``nn.Module`` for vision encoder layer *layer_idx*."""
        ...

    @abstractmethod
    def get_final_norm(self, mt: ModelAndTokenizer) -> nn.Module:
        """Return the final layer-norm applied after the last decoder layer."""
        ...

    @abstractmethod
    def num_decoder_layers(self, mt: ModelAndTokenizer) -> int:
        """Total number of decoder layers."""
        ...

    @abstractmethod
    def num_vision_layers(self, mt: ModelAndTokenizer) -> int:
        """Total number of vision encoder layers."""
        ...

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @abstractmethod
    def generate(self, mt: ModelAndTokenizer, inputs: dict,
                 max_new_tokens: int = 20) -> str:
        """Run greedy generation and return the decoded output string."""
        ...

    # ------------------------------------------------------------------
    # Forward-pass input filtering
    # ------------------------------------------------------------------

    # Keys stored in the inputs dict that are NOT accepted by model.forward().
    # Subclasses can extend this list if they store additional metadata.
    _NON_FORWARD_KEYS: tuple = ("_raw_prompt",)

    def get_forward_inputs(self, inputs: dict) -> dict:
        """Return a copy of *inputs* with metadata-only keys removed.

        Some adapters (e.g. InternVL) store extra keys like ``_raw_prompt``
        in the inputs dict for use by :meth:`generate`, but these keys must
        not be passed to ``model.forward()``.  Call this method before any
        direct ``model(**inputs)`` invocation.
        """
        return {k: v for k, v in inputs.items() if k not in self._NON_FORWARD_KEYS}

    # ------------------------------------------------------------------
    # Image token range (for decoder-level image patching)
    # ------------------------------------------------------------------
    @abstractmethod
    def find_image_token_range(self, mt: ModelAndTokenizer,
                               inputs: dict) -> tuple[int, int]:
        """Return ``(start_idx, end_idx)`` of image tokens within ``input_ids``.

        This is used by decoder-level patching to know which hidden-state
        positions correspond to visual information.
        """
        ...
