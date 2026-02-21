#!/bin/bash
# Helper script to run InternVL experiment in parallel across multiple GPUs
# Usage: ./run_parallel.sh

# Configuration
TOTAL_SAMPLES=1000
NUM_GPUS=3
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))

echo "Running experiment on $NUM_GPUS GPUs"
echo "Total samples: $TOTAL_SAMPLES"
echo "Samples per GPU: ~$SAMPLES_PER_GPU"

# GPU 0: samples 0-333
echo "Starting GPU 0 (samples 0-333)"
CUDA_VISIBLE_DEVICES=0 python ivl_exp.py --mode full_sample --start_idx 0 --end_idx 333 &

# GPU 1: samples 333-666
echo "Starting GPU 1 (samples 333-666)"
CUDA_VISIBLE_DEVICES=1 python ivl_exp.py --mode full_sample --start_idx 333 --end_idx 666 &

# GPU 2: samples 666-1000
echo "Starting GPU 2 (samples 666-1000)"
CUDA_VISIBLE_DEVICES=2 python ivl_exp.py --mode full_sample --start_idx 666 --end_idx 1000 &

echo "All jobs started. Check progress in separate terminals or use 'tail -f ivl_results_*.json'"
echo "To monitor: watch -n 1 'ls -lh ivl_results_*.json'"

# Wait for all background jobs to complete
wait

echo "All jobs completed!"
echo "Merging results..."

# Optional: merge results (you can do this manually or with a Python script)
echo "Results saved in:"
echo "  - ivl_results_0_333.json"
echo "  - ivl_results_333_666.json"
echo "  - ivl_results_666_1000.json"

