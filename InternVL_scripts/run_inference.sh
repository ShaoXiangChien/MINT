#!/bin/bash

# Setup CUDA devices (0 and 1)
export CUDA_VISIBLE_DEVICES=3,4

# Install dependencies if needed (uncomment to run)
# pip install -r requirements.txt
# pip install flash-attn --no-build-isolation

# Run the inference script
python inference_internvl.py

