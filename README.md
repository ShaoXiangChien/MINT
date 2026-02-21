# MINT: Causally Tracing Information Fusion in Multimodal Large Language Models

This repository contains code and resources related to our ICLR 2026 submission that investigates the internal mechanisms of multimodal information fusion in Vision-Language Models (VLMs).

## Overview

Multimodal Large Language Models (MLLMs), especially Vision-Language Models, combine visual and textual signals for tasks such as image captioning, visual question answering, and spatial reasoning. However, the internal fusion process of these modalities within the models remains poorly understood.

We propose **MINT (Multimodal Intervention Tracing)**, a systematic framework for causally mapping the fusion pathways within the decoder layers of VLMs. Our method builds on hidden state patching to trace and intervene in the fusion process, identifying a critical "fusion band" — a decisive layer window where visual and linguistic signals actively fuse.

## Key Contributions

- A principled causal intervention technique adapted for multimodal models.
- Empirical mapping of fusion bands across three representative models: LLaVA-1.5-7B, DeepSeek-VL2-Tiny, and Qwen2-VL-7B.
- Diagnostic linking of fusion points to common model failure modes including hallucinations, spatial errors, and negation mistakes.

## Contents

- Code for implementing the MINT framework, including hidden state patching utilities.
- Experimental setups for probing fusion mechanisms using custom and standard multimodal benchmarks.
- Scripts to reproduce interventions and evaluation metrics like Override Accuracy, Flip Rate, and Failure Depth.
- Detailed documentation on architectures and datasets used.

## Reproducibility

Our code is designed to enable reproducible causal diagnostics for multimodal fusion analyses, facilitating future research on interpretability and robustness of VLMs.
