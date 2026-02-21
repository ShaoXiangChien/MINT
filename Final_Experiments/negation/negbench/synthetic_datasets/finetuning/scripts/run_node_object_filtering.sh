#!/bin/bash

# Pass node number as the first argument
NODE_NUM=$1
base_dir="<CLUSTER_BASE_DIR>"  # Placeholder: Replace with your cluster's base directory
input_file="${base_dir}/csvs/negation_dataset/combined/cc12m_images_extracted_pos_neg.csv"
output_base="${base_dir}/csvs/negation_dataset/cc12m_images_pos_neg_filtered"

# Total rows to process
TOTAL_ROWS=10003877
# Number of jobs (24 jobs total, 8 per node)
NUM_JOBS=24
# Number of rows per job (divide by all jobs, including the last one)
ROWS_PER_JOB=$((TOTAL_ROWS / NUM_JOBS))

# Each node runs 8 jobs, so determine job index based on node number
FIRST_JOB_INDEX=$(((NODE_NUM - 1) * 8))

# Function to calculate start and end index for a job
get_indices () {
    local job_index=$1
    local start_index=$((job_index * ROWS_PER_JOB))
    
    # For the last job, set index_end to -1 (process remaining rows)
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

# Loop over 8 jobs for this node (one for each GPU)
for i in $(seq 0 7); do
    # Calculate the job index for this node
    JOB_INDEX=$((FIRST_JOB_INDEX + i))
    
    # Get start and end indices for the job
    read START_INDEX END_INDEX <<< $(get_indices $JOB_INDEX)
    
    # Output file with start and end indices in the filename
    output_file="${output_base}_${START_INDEX}_${END_INDEX}.csv"
    
    # Run the job on the corresponding GPU
    echo "Running job $JOB_INDEX on GPU $i with rows $START_INDEX to $END_INDEX"
    
    CUDA_VISIBLE_DEVICES=$i python filter_negative_objects.py --input_file $input_file --output_file $output_file --start_index $START_INDEX --end_index $END_INDEX > logs/filter_${JOB_INDEX}.log 2>&1 &
    
    # Wait for 30 seconds before launching the next job
    sleep 30
done

echo "Node $NODE_NUM launched all jobs and exited."
