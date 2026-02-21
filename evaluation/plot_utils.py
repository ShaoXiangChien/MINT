"""Shared plotting utilities and style configuration."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def set_paper_style():
    """Apply a consistent plot style for paper figures."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


# Layer label mappings for each model family
MODEL_LAYERS = {
    "llava": {"total": 32, "step": 2, "label": "LLaVA-1.5-7B"},
    "deepseek": {"total": 27, "step": 2, "label": "DeepSeek-VL2-Tiny"},
    "qwen": {"total": 28, "step": 3, "label": "Qwen2-VL-7B"},
    "internvl": {"total": 32, "step": 3, "label": "InternVL3.5-8B"},
}


def get_layer_ticks(model_name, max_layers=None, step=None):
    """Return tick positions and labels for a model's layer sweep."""
    info = MODEL_LAYERS.get(model_name, {"total": 28, "step": 3})
    total = max_layers or info["total"]
    s = step or info["step"]
    ticks = list(range(0, total, s))
    return ticks, [str(t) for t in ticks]
