#!/usr/bin/env bash
# Run all MINT patching experiments sequentially.
# Adjust --device and --model flags as needed for your hardware.
set -e

DEVICE="cuda:0"
RESULTS="results"
mkdir -p "$RESULTS"

echo "=========================================="
echo "  MINT -- Running All Experiments"
echo "=========================================="

echo ""
echo "[1/6] Object Patching (Vision Encoder)"
for MODEL in llava deepseek qwen; do
    echo "  -> $MODEL"
    python -m experiments.01_object_patching.run_experiment \
        --model "$MODEL" --device "$DEVICE" \
        --output "$RESULTS/${MODEL}_object_patching.json"
done

echo ""
echo "[2/6] Multimodal Fusion (Decoder)"
for MODEL in llava deepseek qwen; do
    echo "  -> $MODEL"
    python -m experiments.02_multimodal_fusion.run_experiment \
        --model "$MODEL" --device "$DEVICE" \
        --output "$RESULTS/${MODEL}_multimodal_fusion.json"
done

echo ""
echo "[3/6] Text Patching (Decoder)"
for MODEL in llava deepseek qwen; do
    echo "  -> $MODEL"
    python -m experiments.03_text_patching.run_experiment \
        --model "$MODEL" --device "$DEVICE" \
        --output "$RESULTS/${MODEL}_text_patching.json"
done

echo ""
echo "[4/6] Global Image Fusion (Decoder)"
for MODEL in llava deepseek qwen internvl; do
    echo "  -> $MODEL"
    python -m experiments.04_global_image_fusion.run_experiment \
        --model "$MODEL" --device "$DEVICE" \
        --output "$RESULTS/${MODEL}_global_image_fusion.json"
done

echo ""
echo "[5/6] Spatial Reasoning (Decoder)"
for MODEL in llava deepseek qwen; do
    echo "  -> $MODEL"
    python -m experiments.05_spatial_reasoning.run_experiment \
        --model "$MODEL" --device "$DEVICE" \
        --dataset data/controlled_images_dataset.json \
        --image_dir data/spatial_images/ \
        --output "$RESULTS/${MODEL}_spatial_reasoning.json"
done

echo ""
echo "[6/6] Negation / SURF (Decoder)"
for MODEL in llava deepseek qwen; do
    echo "  -> $MODEL"
    python -m experiments.06_negation.run_surf_experiment \
        --model "$MODEL" --device "$DEVICE" \
        --test_images_dir data/surf/test_images/ \
        --output "$RESULTS/${MODEL}_negation.json"
done

echo ""
echo "All experiments complete. Results in $RESULTS/"
