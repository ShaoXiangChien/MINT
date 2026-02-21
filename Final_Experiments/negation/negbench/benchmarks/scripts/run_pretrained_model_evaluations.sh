#!/bin/bash

# SLURM Job Submission Script for Pretrained Model Evaluations

# Base Directories
BASE_DIR="/path/to/your/research/project"  # Replace with your base directory
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="$BASE_DIR/models"

# SLURM Configuration
SBATCH_PARTITION="gpu_partition"  # Replace with your cluster's partition name
SBATCH_GPUS=1
SBATCH_CPUS=16
SBATCH_MEM="20gb"
SBATCH_TIME="2:00:00"  # Adjust as needed

# Flag for video evaluations
video=false  # Set to true for video evaluations

# Experiment type: choose between "main" or "template"
experiment_type="main"  # Set this to either "main" or "template"

# Determine evaluation script
if [ "$video" = true ]; then
    sbatch_script="evaluate_videos.sh"
    echo "Submitting video evaluations..."
else
    sbatch_script="evaluate_images.sh"
    echo "Submitting image evaluations..."
fi

# Pretrained model configurations
models=("ViT-B-32" "ViT-L-14")
pretrained_options=("openai" "datacomp_xl_s13b_b90k" "laion400m_e31")

# Submit jobs for pretrained models
for model in "${models[@]}"; do
    for pretrained in "${pretrained_options[@]}"; do
        echo "Submitting job for Model: $model, Pretrained: $pretrained, Experiment: $experiment_type"

        sbatch --job-name="eval_${model}_${pretrained}_${experiment_type}" \
               --partition=$SBATCH_PARTITION \
               --gres=gpu:$SBATCH_GPUS \
               --cpus-per-task=$SBATCH_CPUS \
               --mem=$SBATCH_MEM \
               --time=$SBATCH_TIME \
               --output="$LOGS_DIR/${model}_${pretrained}_${experiment_type}_job_%j.out" \
               --error="$LOGS_DIR/${model}_${pretrained}_${experiment_type}_job_%j.err" \
               --export=ALL,model="$model",pretrained="$pretrained",experiment_type="$experiment_type" \
               $sbatch_script
    done
done
