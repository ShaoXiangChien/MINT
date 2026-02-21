#!/bin/bash

# Set the base directory for data and logs. Users should update this to their directory structure.
IMAGE_METADATA_DIR="PATH_TO_IMAGE_METADATA"
LOGS_DIR="../logs"

# Model and pretrained options
MODEL="ViT-B-32"
MODEL_NAME="ViT_B_32_openai"
PRETRAINED_MODEL="openai"
# If evaluating a pre-trained model that is stored locally, set the path to the model
# MODEL_NAME="NegCLIP"
# MODELS_DIR="PATH_TO_MODELS"
# PRETRAINED_MODEL="$MODELS_DIR/NegCLIP/negclip.pth"

# Dataset paths for images
COCO_MCQ="$IMAGE_METADATA_DIR/COCO_val_mcq_llama3.1_rephrased.csv"
VOC_MCQ="$IMAGE_METADATA_DIR/VOC2007_mcq_llama3.1_rephrased.csv"
COCO_RETRIEVAL="$IMAGE_METADATA_DIR/COCO_val_retrieval.csv"
COCO_NEGATED_RETRIEVAL="$IMAGE_METADATA_DIR/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"

# Dataset paths for videos
VIDEO_METADATA_DIR="PATH_TO_VIDEO_METADATA"
MSRVTT_RETRIEVAL="$VIDEO_METADATA_DIR/msr_vtt_retrieval.csv"
MSRVTT_NEGATED_RETRIEVAL="$VIDEO_METADATA_DIR/msr_vtt_retrieval_rephrased_llama.csv"
MSRVTT_MCQ="$VIDEO_METADATA_DIR/msr_vtt_mcq_rephrased_llama.csv"

# Activate the appropriate environment
source activate clip_negation || conda activate clip_negation

# Set system limits
ulimit -S -n 100000

# Logs directory for this evaluation run
RUN_LOGS_DIR="$LOGS_DIR/evaluation"
mkdir -p "$RUN_LOGS_DIR"

cd ..

# Image Evaluation
echo "Starting Image Evaluation..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model $MODEL \
    --pretrained $PRETRAINED_MODEL \
    --name "image_$MODEL_NAME" \
    --logs=$RUN_LOGS_DIR \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --zeroshot-frequency 1 \
    --imagenet-val="$DATA_DIR/images/imagenet" \
    --coco-mcq=$COCO_MCQ \
    --voc2007-mcq=$VOC_MCQ \
    --coco-retrieval=$COCO_RETRIEVAL \
    --coco-negated-retrieval=$COCO_NEGATED_RETRIEVAL \
    --batch-size=64 \
    --workers=8

# Can also perform video evaluation in a similar way
# The additional command line arguments for video evaluation are:
# --msrvtt-retrieval=$MSRVTT_RETRIEVAL \
# --msrvtt-negated-retrieval=$MSRVTT_NEGATED_RETRIEVAL \
# --msrvtt-mcq=$MSRVTT_MCQ \
# --video \