"""Model adapter registry.

Usage::

    from src.models import get_adapter
    adapter = get_adapter("qwen")
    mt = adapter.load_model("Qwen/Qwen2-VL-7B-Instruct", device="cuda:0")
"""

from .llava import LLaVAAdapter
from .deepseek import DeepSeekAdapter
from .qwen import QwenAdapter
from .internvl import InternVLAdapter

_REGISTRY = {
    "llava": LLaVAAdapter,
    "deepseek": DeepSeekAdapter,
    "qwen": QwenAdapter,
    "internvl": InternVLAdapter,
}


def get_adapter(model_name: str):
    """Return an instantiated adapter for the given model family.

    Args:
        model_name: One of ``"llava"``, ``"deepseek"``, ``"qwen"``, ``"internvl"``.

    Returns:
        A :class:`BaseModelAdapter` subclass instance.
    """
    key = model_name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(_REGISTRY)}"
        )
    return _REGISTRY[key]()
