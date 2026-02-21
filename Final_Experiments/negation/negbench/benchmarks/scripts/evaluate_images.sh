#!/bin/bash

# SLURM Image Evaluation Script

# SLURM Directives
#SBATCH --job-name=eval_images
#SBATCH --partition=gpu_partition  # Replace with your cluster's partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20gb
#SBATCH --time=2:00:00  # Adjust as needed
#SBATCH --output=jobs/image_job_%j.out
#SBATCH --error=jobs/image_job_%j.err

# Print input variables
echo "Model: $model"
echo "Pretrained: $pretrained"
echo "Experiment Type: $experiment_type"

# Activate the environment
source activate clip_negation || conda activate clip_negation
ulimit -S -n 100000

# Select paths based on experiment type
if [ "$experiment_type" = "main" ]; then
    logs="$BASE_DIR/logs/main_results"
    coco_mcq="$BASE_DIR/data/images/COCO_val_mcq_llama3.1_rephrased.csv"
    voc_mcq="$BASE_DIR/data/images/VOC2007_mcq_llama3.1_rephrased.csv"
    synthetic_mcq="$BASE_DIR/data/images/synthetic_mcq_llama3.1_rephrased.csv"
    coco_negated_retrieval="$BASE_DIR/data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    coco_retrieval="$BASE_DIR/data/images/COCO_val_retrieval.csv"
    synthetic_negated_retrieval="$BASE_DIR/data/images/synthetic_retrieval_v2.csv"
    synthetic_retrieval="$BASE_DIR/data/images/synthetic_retrieval_v1.csv"
elif [ "$experiment_type" = "template" ]; then
    logs="$BASE_DIR/logs/template_results"
    coco_mcq="$BASE_DIR/data/images/COCO_val_mcq.csv"
    voc_mcq="$BASE_DIR/data/images/VOC2007_mcq.csv"
    synthetic_mcq="$BASE_DIR/data/images/synthetic_mcq.csv"
    coco_negated_retrieval="$BASE_DIR/data/images/COCO_val_negated_retrieval.csv"
    coco_retrieval="$BASE_DIR/data/images/COCO_val_retrieval.csv"
    synthetic_negated_retrieval="$BASE_DIR/data/images/synthetic_retrieval_v2.csv"
    synthetic_retrieval="$BASE_DIR/data/images/synthetic_retrieval_v1.csv"
else
    echo "Invalid experiment type: $experiment_type"
    exit 1
fi

# Run Image Evaluation
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model $model \
    --pretrained $pretrained \
    --logs=$logs \
    --zeroshot-frequency 1 \
    --imagenet-val="$BASE_DIR/data/images/imagenet" \
    --coco-mcq=$coco_mcq \
    --voc2007-mcq=$voc_mcq \
    --synthetic-mcq=$synthetic_mcq \
    --coco-retrieval=$coco_retrieval \
    --coco-negated-retrieval=$coco_negated_retrieval \
    --synthetic-retrieval=$synthetic_retrieval \
    --synthetic-negated-retrieval=$synthetic_negated_retrieval \
    --batch-size=64 \
    --workers=8
