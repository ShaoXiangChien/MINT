#!/bin/bash

# SLURM Video Evaluation Script

# SLURM Directives
#SBATCH --job-name=eval_videos
#SBATCH --partition=gpu_partition  # Replace with your cluster's partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50gb
#SBATCH --time=7-00:00:00  # Time limit: 7 days
#SBATCH --output=jobs/video_job_%j.out
#SBATCH --error=jobs/video_job_%j.err

# Print input variables
echo "Model: $model"
echo "Pretrained: $pretrained"

# Activate the environment
source activate clip_negation || conda activate clip_negation
ulimit -S -n 100000

# Logs Directory
logs="$BASE_DIR/logs/eval_videos"

# Dataset paths for video evaluations
msrvtt_retrieval="$BASE_DIR/data/videos/msr_vtt_retrieval.csv"
msrvtt_negated_retrieval="$BASE_DIR/data/videos/msr_vtt_retrieval_rephrased_llama.csv"
msrvtt_mcq="$BASE_DIR/data/videos/msr_vtt_mcq_rephrased_llama.csv"

# Run Video Evaluation
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model $model \
    --pretrained $pretrained \
    --logs=$logs \
    --zeroshot-frequency 1 \
    --imagenet-val="$BASE_DIR/data/images/imagenet" \
    --msrvtt-retrieval=$msrvtt_retrieval \
    --msrvtt-negated-retrieval=$msrvtt_negated_retrieval \
    --msrvtt-mcq=$msrvtt_mcq \
    --video \
    --batch-size=64 \
    --workers=8
