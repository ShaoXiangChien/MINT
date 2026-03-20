"""Model adapter registry.

Adapters are loaded lazily: the underlying model package is only imported
when ``get_adapter`` is called for that specific model. This means you only
need to install the dependencies for the model(s) you actually want to run.

Usage::

    from src.models import get_adapter
    adapter = get_adapter("llava_onevision")
    mt = adapter.load_model("llava-hf/llava-onevision-qwen2-7b-ov-hf", device="cuda:0")
"""

# Map of registry key -> (module path, class name)
# The module is imported on demand inside get_adapter().
_REGISTRY = {
    "llava":            ("src.models.llava",            "LLaVAAdapter"),
    "deepseek":         ("src.models.deepseek",         "DeepSeekAdapter"),
    "qwen":             ("src.models.qwen",             "QwenAdapter"),
    "internvl":         ("src.models.internvl",         "InternVLAdapter"),
    "llava_onevision":  ("src.models.llava_onevision",  "LLaVAOneVisionAdapter"),
}


def get_adapter(model_name: str):
    """Return an instantiated adapter for the given model family.

    Only the adapter for the requested model is imported, so missing
    optional dependencies (e.g. the ``llava`` source package) will not
    cause errors when using a different model.

    Args:
        model_name: One of ``"llava"``, ``"deepseek"``, ``"qwen"``,
                    ``"internvl"``, ``"llava_onevision"``.

    Returns:
        A :class:`BaseModelAdapter` subclass instance.

    Raises:
        ValueError: If ``model_name`` is not in the registry.
        ImportError: If the required package for the requested model is
                     not installed, with a hint on how to install it.
    """
    key = model_name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(_REGISTRY)}"
        )

    module_path, class_name = _REGISTRY[key]

    try:
        import importlib
        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
    except ImportError as e:
        _install_hints = {
            "llava": (
                "The LLaVA-1.5 adapter requires the 'llava' package, which is not on PyPI.\n"
                "Install it from source:\n"
                "  git clone https://github.com/haotian-liu/LLaVA.git\n"
                "  pip install -e ./LLaVA"
            ),
            "deepseek": "pip install -r requirements/deepseek.txt",
            "qwen":     "pip install -r requirements/qwen.txt",
            "internvl": "pip install -r requirements/internvl.txt",
            "llava_onevision": "pip install -r requirements/llava_onevision.txt",
        }
        hint = _install_hints.get(key, "Check requirements/base.txt for installation instructions.")
        raise ImportError(
            f"Could not load adapter for '{key}': {e}\n\n"
            f"To install the required dependencies:\n  {hint}"
        ) from e

    return adapter_cls()
