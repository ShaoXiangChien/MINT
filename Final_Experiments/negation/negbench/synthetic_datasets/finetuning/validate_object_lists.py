"""
validate_object_lists.py: Ensures that the extracted objects and negative objects in a dataset are valid lists of objects without 
additional words, phrases, or explanations. This script is used to clean and validate datasets for downstream processing.

### Input CSV Requirements:
The input CSV file must contain the following columns:
- `extracted_objects`: A list of extracted objects for each row.
- `negative_objects`: A list of negative objects for each row.
- `filepath`: (Optional) Path to the associated image or data file for tracking filtered rows.

### Output CSVs:
1. A validated dataset (`*_verified.csv`) containing rows where both `extracted_objects` and `negative_objects` are valid.
2. A filtered-out dataset (`*_filtered_out.csv`) listing the rows that failed validation, including the `filepath` for reference.

### Example Usage:
```bash
python validate_object_lists.py \
    --input_file path/to/input.csv \
    --index_start 0 \
    --index_end 100 \
    --output_base path/to/output \
    --model llama3.1
```

### Parameters:
- `--input_file`: Path to the input CSV file.
- `--index_start`: The starting index of rows to process (inclusive). Default is 0.
- `--index_end`: The ending index of rows to process (exclusive). Default is -1 (process all rows).
- `--output_base`: Base name for the output files.
- `--model`: The LLM model to use for validation (`mixtral` or `llama3.1`).

### Key Steps:
1. **Generate Prompts**:
   - Prompts are created for both `extracted_objects` and `negative_objects` using pre-defined examples.
   - Each prompt is sent to the LLM for validation.

2. **LLM Validation**:
   - The LLM outputs either "Yes" or "No" for each prompt to indicate if the list meets the criteria.

3. **Filter Rows**:
   - Rows where both `extracted_objects` and `negative_objects` are valid ("Yes") are retained.
   - Rows failing validation are saved separately for debugging or manual review.

### Notes:
1. **Batch Processing**:
   - Use `--index_start` and `--index_end` to process large datasets in smaller chunks.

2. **Examples**:
   - Example prompts guide the LLM's understanding of valid and invalid formats.

3. **Edge Cases**:
   - Rows with malformed `extracted_objects` or `negative_objects` will fail validation.
   - Filtered-out rows are saved separately for easier inspection.

4. **LLM Initialization**:
   - Supports `mixtral` and `llama3.1` with automatic fallback for parallelism.

5. **Output Naming**:
   - Files are named based on the processed range (`*_verified.csv` and `*_filtered_out.csv`).

### Output Summary:
- Total rows processed.
- Rows retained after validation.
- Rows filtered out for failing validation.
"""

import argparse
import pandas as pd
from vllm import LLM, SamplingParams
import time
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Validates whether the extracted objects and negative objects include only a list of objects with no additional words or explanations.")
    parser.add_argument("--input_file", type=str, required=True, help="The input CSV file containing extracted objects and negative objects.")
    parser.add_argument("--index_start", type=int, default=0, help="The starting index of rows to process (inclusive).")
    parser.add_argument("--index_end", type=int, default=-1, help="The ending index of rows to process (exclusive).")
    parser.add_argument("--output_base", type=str, default="output", help="The base name for the output CSV file.")
    parser.add_argument("--model", type=str, default="llama3.1", choices=["mixtral", "llama3.1"],
                        help="The LLM model to use. Options: 'mixtral' or 'llama3.1'.")
    return parser.parse_args()

def generate_prompt(objects_value, examples):
    """
    Generates a prompt for the LLM to determine if the given list of objects
    includes only objects with no additional words or explanations.

    Args:
        objects_value (str): The string representation of the list of objects to be evaluated.
        examples (str): Example prompts and responses to guide the LLM's output.

    Returns:
        str: A formatted prompt string to be used as input to the LLM.
    """
    prompt_template = f"""Answer with "Yes" or "No" only.
Does the following include a list of objects with no additional words or explanations?

{examples}

{objects_value}
"""
    return prompt_template

def process_llm_output(output_text):
    """
    Processes the LLM output to extract the answer ('Yes' or 'No').

    Args:
        output_text (str): The text output by the LLM.

    Returns:
        str: 'Yes' or 'No', depending on the LLM's answer, or None if unclear.
    """
    # Strip leading and trailing whitespace
    output_text = output_text.strip()
    # Convert to lower case
    output_lower = output_text.lower()
    if output_lower.startswith('yes'):
        return 'Yes'
    elif output_lower.startswith('no'):
        return 'No'
    else:
        return None  # Could not determine

def initialize_llm(model_name):
    """
    Initializes the LLM model and sets up the sampling parameters.

    Parameters:
        model_name (str): The full model name to be used for the LLM.

    Returns:
        llm (LLM): The initialized LLM object.
        sampling_params (SamplingParams): The configured sampling parameters.
    """
    try:
        print(f"Trying to initialize LLM with tensor_parallel_size=4...")
        llm = LLM(model=model_name, tensor_parallel_size=4)
    except Exception as e:
        print(f"Failed to initialize LLM with tensor_parallel_size=4: {e}")
        print(f"Initializing LLM with tensor_parallel_size=2...")
        llm = LLM(model=model_name, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, stop=["\n"])
    return llm, sampling_params

def process_validation_task(df, model_name, args):
    """
    Processes the task by using LLM to validate if the extracted objects and negative objects include only a list of objects
    with no additional words or explanations.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data, including 'extracted_objects' and 'negative_objects' columns.
        model_name (str): The full model name to be used for the LLM.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple:
            pd.DataFrame: A DataFrame containing only the rows where both 'extracted_objects_valid' 
                          and 'negative_objects_valid' are 'Yes'.
            pd.DataFrame: A DataFrame containing the rows that were filtered out, i.e., where either
                          'extracted_objects_valid' or 'negative_objects_valid' was not 'Yes'. This DataFrame
                          includes the 'filepath' column for identifying the filtered-out rows.
    """
    # Initialize new columns to store the LLM answers
    df["extracted_objects_valid"] = None
    df["negative_objects_valid"] = None

    prompts = []  # Collect all prompts to batch together
    indices = []  # Track the index for each prompt
    columns = []  # Track which column ('extracted_objects_valid' or 'negative_objects_valid') the prompt is for

    # Prepare the examples matching the format of the objects in the CSV
    examples = """['room', 'kitchen', 'foyer']
Yes

"['bathroom tiles', 'furniture', 'building blueprint', 'paint bucket', 'rug\\n# Task\\nCaption: The Great Wave off Kanagawa', 'a famous woodblock print by Japanese artist Hokusai\\nRelevant Objects: [wave', 'print', 'artist', 'hokusai]\\nOutput: easel', 'palette', 'woodblock', 'painting', 'Japan', 'museum', 'art gallery', 'Tokyo', 'ocean\\n Corrected Output: easel', 'Tokyo\\nNote: The woodblock term is already part of the caption', 'so it shouldn't be included as a negative object. Here is the next task:']"
No

"['(assuming <PERSON> is a placeholder for a name', 'the focus of the caption seems to be the wedding - but the only known photog and related wedding objects are)\\nphotographer', 'wedding']"
No

"['palm trees', 'sailboat', 'beach towel', 'surfboard', 'beach umbrella']"
Yes

"['director', 'game', 'Netflix', 'series']"
Yes
"""

    # Start generating prompts for all the rows
    for idx in tqdm(df.index, desc="Generating prompts"):
        # For extracted_objects
        extracted_objects = df.loc[idx, "extracted_objects"]
        prompt_extracted = generate_prompt(extracted_objects, examples)
        prompts.append(prompt_extracted)
        indices.append(idx)
        columns.append('extracted_objects_valid')

        # For negative_objects
        negative_objects = df.loc[idx, "negative_objects"]
        prompt_negative = generate_prompt(negative_objects, examples)
        prompts.append(prompt_negative)
        indices.append(idx)
        columns.append('negative_objects_valid')

    # Initialize the LLM and sampling parameters
    llm, sampling_params = initialize_llm(model_name)

    # Send all prompts in a single batch to the LLM
    start_time = time.time()
    print("Processing prompts with LLM...")
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    print(f"Processing time (for {len(prompts)} prompts): {end_time - start_time} seconds")

    # Process the LLM outputs and store the answers
    for idx, output in enumerate(outputs):
        try:
            generated_text = output.outputs[0].text
            llm_answer = process_llm_output(generated_text)

            # Get the corresponding DataFrame index and column
            df_index = indices[idx]
            column_name = columns[idx]

            # Store the LLM answer
            df.loc[df_index, column_name] = llm_answer

        except Exception as e:
            print(f"Error processing output for index {df_index}, column {column_name}: {e}")
            df.loc[df_index, column_name] = None

    # Remove rows where either answer was not 'Yes'
    initial_len = len(df)
    df_filtered = df[(df["extracted_objects_valid"] == 'Yes') & (df["negative_objects_valid"] == 'Yes')].copy()
    df_filtered_out = df[~((df["extracted_objects_valid"] == 'Yes') & (df["negative_objects_valid"] == 'Yes'))].copy()

    filtered_len = len(df_filtered)
    print(f"Filtered out {initial_len - filtered_len} rows where the answer was not 'Yes' for either column.")

    # Drop the validation columns as they're no longer needed
    df_filtered.drop(columns=["extracted_objects_valid", "negative_objects_valid"], inplace=True)
    df_filtered_out.drop(columns=["extracted_objects_valid", "negative_objects_valid"], inplace=True)

    return df_filtered, df_filtered_out

def main(args):
    # Map model aliases to full model names
    model_mapping = {
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama3.1": "meta-llama/Meta-Llama-3.1-70B-Instruct"
    }

    model_name = model_mapping[args.model]  # Use alias to get full model name

    # Read the input CSV
    df = pd.read_csv(args.input_file)

    # Ensure required columns exist
    required_columns = ['extracted_objects', 'negative_objects']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The input CSV must contain a '{col}' column.")

    # Set index_end to the length of the DataFrame if it's -1 or exceeds the length of the DataFrame
    if args.index_end == -1 or args.index_end > len(df):
        args.index_end = len(df)

    # Subset the DataFrame to only the rows specified by index_start and index_end
    df_subset = df.iloc[args.index_start:args.index_end].copy()

    # Process the validation task
    df_subset_filtered, df_subset_filtered_out = process_validation_task(df_subset, model_name, args)

    # Generate output file name based on the processed range
    if args.index_start == 0 and args.index_end == len(df):
        output_file_name = f"{args.output_base}_verified.csv"
        filtered_out_file_name = f"{args.output_base}_filtered_out.csv"
    else:
        output_file_name = f"{args.output_base}_{args.index_start}_{args.index_end}_filtered.csv"
        filtered_out_file_name = f"{args.output_base}_{args.index_start}_{args.index_end}_filtered_out.csv"

    # Save the modified DataFrame with the new negative captions
    df_subset_filtered.to_csv(output_file_name, index=False)

    # Save the file paths of filtered-out rows
    df_subset_filtered_out[["filepath"]].to_csv(filtered_out_file_name, index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)
