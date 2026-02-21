# MINT: Multimodal INformation Tracing

Code and resources for causally tracing information fusion in Multimodal Large Language Models (MLLMs).

## Overview

Vision-Language Models (VLMs) combine visual and textual signals for tasks such as visual question answering, image captioning, and spatial reasoning.  However, the internal fusion process between modalities remains poorly understood.

**MINT** is a systematic causal-intervention framework that maps the *fusion pathways* within VLM decoder layers.  Using hidden-state patching, MINT identifies **fusion bands** -- critical layer windows where visual and linguistic representations actively merge -- and links these bands to common failure modes including hallucinations, spatial errors, and negation mistakes.

### Key Contributions

- A principled causal-intervention technique (hidden-state patching) adapted for multimodal architectures.
- Empirical mapping of fusion bands across four representative models: **LLaVA-1.5-7B**, **DeepSeek-VL2-Tiny**, **Qwen2-VL-7B**, and **InternVL3.5-8B**.
- Diagnostic linking of fusion points to failure modes, with layer-specific LoRA fine-tuning as a targeted intervention.

## Repository Structure

```
MINT/
├── src/                            # Core library
│   ├── models/                     # Model adapters (LLaVA, DeepSeek, Qwen, InternVL)
│   ├── patching/                   # Patching engine (hooks, decoder & vision patching)
│   └── utils/                      # Shared token utilities
├── experiments/                    # Experiment scripts (one directory per experiment)
│   ├── 01_object_patching/         # Vision-encoder object patching
│   ├── 02_multimodal_fusion/       # Decoder-level multimodal patching
│   ├── 03_text_patching/           # Text-only hidden-state patching
│   ├── 04_global_image_fusion/     # Global image-token patching
│   ├── 05_spatial_reasoning/       # Spatial relationship patching
│   └── 06_negation/                # Negation understanding (SURF + LoRA)
├── evaluation/                     # Analysis & visualisation
│   ├── bootstrap_ci.py             # Bootstrap confidence intervals
│   ├── report_bootstrap.py         # Tables, charts, heatmaps
│   └── generate_paper_figures.py   # Paper-ready figure generation
├── data/                           # Dataset loading & download scripts
├── configs/                        # LoRA adapter configurations
├── scripts/                        # Shell scripts for batch execution
├── requirements.txt
└── .gitignore
```

## Environment Setup

### Prerequisites

- Python >= 3.10
- CUDA >= 11.8 (for GPU inference)
- ~40 GB GPU memory recommended (A100 or equivalent)

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd MINT

# Create a conda environment (recommended)
conda create -n mint python=3.10 -y
conda activate mint

# Install dependencies
pip install -r requirements.txt

# LLaVA must be installed separately (it is not on PyPI)
# Follow: https://github.com/haotian-liu/LLaVA#install
pip install -e /path/to/LLaVA
```

## Data Preparation

### COCO 2017 Validation Set

Used by experiments 01--04.  Download with:

```bash
python -m data.download_coco --output_dir data/
```

### Pre-processed Sample Dataset

Place the HuggingFace `Dataset` (saved via `save_to_disk`) in `data/full_sample/` and the category mapping in `data/instances_category_map.json`.  See `data/README.md` for details.

### Spatial Relationship Dataset

Place `controlled_images_dataset.json` and the corresponding images in `data/`.  See experiment 05 for generation instructions.

### NegBench (External)

The NegBench negation benchmark is an external dependency.  Follow the upstream instructions to obtain the dataset:

```bash
pip install negbench
```

## Reproducing Key Results

All experiments use a unified `--model {llava,deepseek,qwen,internvl}` interface.

### Quick Start: Run a Single Experiment

```bash
# Example: Global image fusion with Qwen2-VL
python -m experiments.04_global_image_fusion.run_experiment \
    --model qwen \
    --device cuda:0 \
    --output results/qwen_global_image_fusion.json
```

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

### Run All Analyses

```bash
bash scripts/run_all_analyses.sh
```

### Script-to-Paper Mapping

| Paper Reference | Script / Command |
|---|---|
| **Table 3** -- Override Accuracy (Bootstrap) | `python -m evaluation.bootstrap_ci --results_dir results/ --out_dir analysis/` |
| **Figure 3** -- Fusion Band Heatmaps | `python -m evaluation.generate_paper_figures --experiment_dirs analysis/ --out_dir figures/` |
| **Figure 4** -- Target-Layer Comparison | `python -m evaluation.report_bootstrap --summary analysis/bootstrap_summary.json --out_dir figures/` |
| **Table 5** -- LoRA Intervention Results | `python -m experiments.06_negation.lora_evaluation --test_data data/test.jsonl --output results/lora_eval.json` |

### Individual Experiments

| Experiment | Description | Command |
|---|---|---|
| 01 Object Patching | Patches vision-encoder object embeddings | `python -m experiments.01_object_patching.run_experiment --model MODEL ...` |
| 02 Multimodal Fusion | Patches decoder image-token hidden states | `python -m experiments.02_multimodal_fusion.run_experiment --model MODEL ...` |
| 03 Text Patching | Patches decoder text-token hidden states | `python -m experiments.03_text_patching.run_experiment --model MODEL ...` |
| 04 Global Image Fusion | Patches all decoder image tokens | `python -m experiments.04_global_image_fusion.run_experiment --model MODEL ...` |
| 05 Spatial Reasoning | Patches for left/right spatial understanding | `python -m experiments.05_spatial_reasoning.run_experiment --model MODEL ...` |
| 06 Negation (SURF) | Patches for negation understanding | `python -m experiments.06_negation.run_surf_experiment --model MODEL ...` |
| 06 Negation (LoRA) | Layer-specific LoRA fine-tuning | `python -m experiments.06_negation.lora_training ...` |

## Architecture

The codebase follows a model-adapter pattern:

- **`src/models/`** provides a uniform interface (`BaseModelAdapter`) for loading models, preparing inputs, and accessing architecture internals (decoder layers, vision layers).  Each VLM family has a concrete adapter.
- **`src/patching/`** implements the two-pass causal intervention protocol:
  1. *Source pass*: run the model and cache hidden states at a chosen layer.
  2. *Target pass*: re-run the model while replacing selected positions with cached states, then generate.
- **`experiments/`** contain self-contained scripts that combine an adapter with the patching engine to sweep over layer pairs.

## Citation

```
@inproceedings{anonymous2026mint,
  title     = {MINT: Causally Tracing Information Fusion in Multimodal Large Language Models},
  author    = {Anonymous},
  booktitle = {Proceedings of ACL 2026},
  year      = {2026},
}
```

## License

This project is released under the Apache 2.0 License.
The patching utilities are adapted from [Patchscopes](https://github.com/google-research/patchscopes) (Apache 2.0).
