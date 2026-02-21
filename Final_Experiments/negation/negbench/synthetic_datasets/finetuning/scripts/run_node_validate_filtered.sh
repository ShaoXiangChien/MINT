#!/bin/bash

# Pass node number as the first argument
NODE_NUM=$1
base_dir="<CLUSTER_BASE_DIR>"  # Placeholder: Replace with your cluster's base directory
input_file="${base_dir}/csvs/negation_dataset/combined/cc12m_images_pos_neg_filtered.csv"
output_base="${base_dir}/csvs/negation_dataset/validated/cc12m_images_pos_neg"

# Total rows to process
TOTAL_ROWS=10003877
# Number of jobs (16 jobs total, 2 per node)
NUM_JOBS=16
# Number of rows per job (divide by all jobs, including the last one)
ROWS_PER_JOB=$((TOTAL_ROWS / NUM_JOBS))

# Each node runs two jobs, so determine job index based on node number
FIRST_JOB_INDEX=$(((NODE_NUM - 1) * 2))
SECOND_JOB_INDEX=$((FIRST_JOB_INDEX + 1))

# Function to calculate start and end index for a job
get_indices () {
    local job_index=$1
    local start_index=$((job_index * ROWS_PER_JOB))
    
    # For the last job (16th), set index_end to -1 (process remaining rows)
    if [ "$job_index" -eq $((NUM_JOBS - 1)) ]; then
        local end_index=-1
    else
        local end_index=$(((job_index + 1) * ROWS_PER_JOB))
    fi
    
    echo "$start_index $end_index"
}

# Ensure you are in the correct directory for the Python script
cd ..  # Move up to the location of the Python scripts, adjust if necessary

source <PATH_TO_CONDA>/etc/profile.d/conda.sh  # Placeholder: Replace with your conda installation path
conda activate negbench
export TMPDIR="<CLUSTER_TMP_DIR>"  # Placeholder: Replace with your cluster's temporary directory
export PIP_CACHE_DIR="<CLUSTER_PIP_CACHE_DIR>"  # Placeholder: Replace with your cluster's pip cache directory
export HF_HOME="<CLUSTER_HF_HOME>"  # Placeholder: Replace with your Hugging Face home directory
export HF_TOKEN="<HF_ACCESS_TOKEN>"  # Placeholder: Replace with your Hugging Face access token

# First job (GPUs 0,1,2,3)
read START_INDEX END_INDEX <<< $(get_indices $FIRST_JOB_INDEX)
echo "Running job $FIRST_JOB_INDEX on GPUs 0,1,2,3 with rows $START_INDEX to $END_INDEX"

CUDA_VISIBLE_DEVICES=0,1,2,3 python validate_object_lists.py --model llama3.1 --input_file $input_file --output_base $output_base --index_start $START_INDEX --index_end $END_INDEX > logs/validate_${FIRST_JOB_INDEX}.log 2>&1 &

# Second job (GPUs 4,5,6,7)
read START_INDEX END_INDEX <<< $(get_indices $SECOND_JOB_INDEX)
echo "Running job $SECOND_JOB_INDEX on GPUs 4,5,6,7 with rows $START_INDEX to $END_INDEX"

CUDA_VISIBLE_DEVICES=4,5,6,7 python validate_object_lists.py --model llama3.1 --input_file $input_file --output_base $output_base --index_start $START_INDEX --index_end $END_INDEX > logs/validate_${SECOND_JOB_INDEX}.log 2>&1 &

echo "Node $NODE_NUM launched both jobs and exited."
