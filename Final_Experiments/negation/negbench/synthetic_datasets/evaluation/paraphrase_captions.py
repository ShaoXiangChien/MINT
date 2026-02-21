"""
This script processes a CSV file containing captions and paraphrases them using a large language model (LLM).
The rephrased captions are saved to a new CSV file. It supports both retrieval and multiple-choice question (MCQ) tasks.

### Functionality:
1. **Input CSV File**:
   - For retrieval tasks:
     - `captions`: A string representation of a list of captions (e.g., `["caption1", "caption2"]`).
   - For MCQ tasks:
     - `caption_0` to `caption_3`: Four answer choices for each row.

2. **Output CSV File**:
   - The output file includes the original columns, but with updated captions that have been rephrased.

3. **Options**:
   - Use the `--task` argument to specify the task:
     - `retrieval`: Paraphrases a list of captions in the `captions` column.
     - `mcq`: Paraphrases individual captions in columns `caption_0` to `caption_3`.
   - The `--use_affirmation_negation_guideline` flag ensures the rephrased captions retain the original negation or affirmation flow.

4. **Models**:
   - `mixtral`: Uses the "mistralai/Mixtral-8x7B-Instruct-v0.1" model.
   - `llama3.1`: Uses the "meta-llama/Meta-Llama-3.1-70B-Instruct" model.

### Example Usage:
#### Paraphrase Retrieval Captions:
```bash
python paraphrase_captions.py \
    --model mixtral \
    --task retrieval \
    --input_file data/COCO_val_negated_retrieval.csv
```
#### Paraphrase MCQ Captions:
```bash
python paraphrase_captions.py \
    --model llama3.1 \
    --task mcq \
    --input_file data/COCO_val_mcq.csv
```
#### Preserve Affirmation/Negation Order:
```bash
python paraphrase_captions.py \
    --model mixtral \
    --task retrieval \
    --input_file data/COCO_val_negated_retrieval.csv \
    --use_affirmation_negation_guideline
```

### Notes:
- The `--index_start` and `--index_end` arguments allow processing of specific rows in the input file.
  - Example: `--index_start 0 --index_end 100` processes rows 0–99.
  - Default is to process the entire file.
- The output file is saved with a suffix indicating the model and processed range (e.g., `input_file_mixtral_rephrased.csv`).
- Ensure the input CSV is correctly formatted based on the task type.
"""

import argparse
import pandas as pd
from vllm import LLM, SamplingParams
import time
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Paraphrase captions using a large language model.")
    parser.add_argument("--input_file", type=str, help="The input CSV file containing captions to paraphrase.")
    parser.add_argument("--index_start", type=int, default=0, help="The starting index of captions to process.")
    parser.add_argument("--index_end", type=int, default=None, help="The ending index of captions to process.")
    parser.add_argument("--model", type=str, default="mixtral", choices=["mixtral", "llama3.1"], 
                        help="The LLM model to use for paraphrasing. Options: 'mixtral' or 'llama3.1'.")
    parser.add_argument("--task", type=str, choices=['retrieval', 'mcq'], required=True, 
                        help="Specify the task: 'retrieval' or 'mcq'")
    parser.add_argument('--use_affirmation_negation_guideline', action='store_true', 
                        help="Include affirmation/negation guideline in paraphrasing task.") # if passed, it is true, else false
    return parser.parse_args()


retrieval_examples = """Original Caption: "A table with pies being made and a person standing near a wall with pots and pans hanging on the wall. There is no fork in the image."
Rephrased Caption: "A table where pies are being prepared and a person stands near a wall with hanging pots and pans, but there is no fork in sight."

Original Caption: "There is no handbag in the image. A man on a skateboard performs a trick at the skate park."
Rephrased Caption: "No handbag is present in the image; instead, a man on a skateboard is performing a trick at the skate park."

Original Caption: "There is no fork in the image. A person standing by a stove in a kitchen."
Rephrased Caption: "No fork is visible in the image, only a person standing by a stove in the kitchen."

Original Caption: "There is no cup in the image. The dining table near the kitchen has a bowl of fruit on it."
Rephrased Caption: "No cup can be seen in the image; instead, the dining table near the kitchen holds a bowl of fruit."

Original Caption: "a homeless man holding a cup and standing next to a shopping cart on a street There is no car in the image."
Rephrased Caption: "A homeless man stands on the street, holding a cup and next to a shopping cart, with no car present in the image." """

mcq_examples = """Original Caption: "This image does not include car."
Rephrased Caption: "There is no car included in this image."

Original Caption: "This image includes sink and spoon."
Rephrased Caption: "This image features a sink and a spoon."

Original Caption: "This image includes fork."
Rephrased Caption: "This image shows a fork."

Original Caption: "This image includes lion but not person."
Rephrased Caption: "This image features a lion, but no person is present."

Original Caption: "This image includes oven but not bottle."
Rephrased Caption: "This image contains an oven but no bottle." """

def generate_prompt(caption, examples, use_affirmation_negation_guideline):
    """
    Generates a prompt for rephrasing a caption.

    Args:
        caption (str): The original caption to be rephrased.
        examples (str): Example prompts and responses to guide the LLM's output.
        use_affirmation_negation_guideline (bool): Flag indicating whether to include the affirmation/negation guideline.

    Returns:
        str: A formatted prompt string to be used as input to the LLM.
    """
    
    guidelines = """
1. Do not introduce any new objects.
2. Keep the captions concise and clear, while preserving the original meaning.
3. Only output the rephrased caption without additional text or explanations.
"""
    
    if use_affirmation_negation_guideline:
        guidelines = """
1. Do not introduce any new objects.
2. If the original caption starts with the affirmation, start with affirmation. If the original caption starts with negation, start with negation. In either way, make it flow more naturally.
3. Keep the captions concise and clear, while preserving the original meaning.
4. Only output the rephrased caption without additional text or explanations.
"""

    prompt_template = f"""You will be given a caption that describes the presence of certain objects, the absence of some objects, or both presence and absence. Your task is to rephrase the caption to improve its flow and make it more engaging. While rephrasing, please follow these guidelines:

{guidelines}

{examples}

Original Caption: "{caption}"
Rephrased Caption: """
    
    return prompt_template

def process_llm_output(output_text):
    """
    Processes the LLM output to extract the rephrased caption.

    The function accounts for different formats the LLM might use, such as:
    - Directly outputting the caption.
    - Enclosing the caption in quotation marks.
    - Preceding the caption with "Rephrased Caption: ".
    - Outputting the caption with additional leading or trailing spaces.
    - Including additional lines with irrelevant content after the caption.

    Args:
        output_text (str): The text output by the LLM.

    Returns:
        str: The cleaned and extracted caption.

    Example usage:
        >>> process_llm_output('Rephrased Caption: "This image shows a cat."\nOther content here.')
        'This image shows a cat.'
        
        >>> process_llm_output('"A dog is sitting on the grass."\nNote: The grass is green.')
        'A dog is sitting on the grass.'

        >>> process_llm_output('This is a simple caption.')
        'This is a simple caption.'

        >>> process_llm_output('Rephrased Caption: This is another caption.\nFurther explanation follows.')
        'This is another caption.'
    """
    # Trim any leading or trailing whitespace
    output_text = output_text.strip()

    # Extract the first line, assuming it contains the relevant caption
    output_text = output_text.split('\n', 1)[0].strip()
    
    # If the output starts with 'Rephrased Caption:', remove this prefix
    if output_text.startswith('Rephrased Caption:'):
        output_text = output_text[len('Rephrased Caption:'):].strip()

    # While the output is enclosed in quotation marks, remove them
    while output_text.startswith('"') and output_text.endswith('"'):
        output_text = output_text[1:-1]

    return output_text

def initialize_llm(model_name):
    """
    Initializes the LLM model and sets up the sampling parameters.

    Parameters:
        model_name (str): The full model name to be used for the LLM.

    Returns:
        llm (LLM): The initialized LLM object.
        sampling_params (SamplingParams): The configured sampling parameters.
    """
    if model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct":
        # use 4 gpus for llama3.1
        llm = LLM(model=model_name, tensor_parallel_size=4)
    else:
        # use 2 gpus for mixtral
        llm = LLM(model=model_name, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=900, stop=["\n\n"])
    return llm, sampling_params

def process_retrieval_task(df, model_name, args, retrieval_examples, use_affirmation_negation_guideline):
    """
    Processes the 'retrieval' task by paraphrasing captions in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        model_name (str): The full model name to be used for the LLM.
        args (argparse.Namespace): Parsed command-line arguments.
        retrieval_examples (list): Examples to be used in generating prompts for retrieval task.
        use_affirmation_negation_guideline (bool): Flag indicating whether to include the affirmation/negation guideline.
    """
    llm, sampling_params = initialize_llm(model_name)
    
    captions_lists = df.iloc[args.index_start:args.index_end]["captions"].apply(eval).tolist()
    
    rephrased_captions_lists = []
    start_time = time.time()
    
    for captions in tqdm(captions_lists, desc="Paraphrasing captions"):
        prompts = [generate_prompt(caption, retrieval_examples, use_affirmation_negation_guideline) for caption in captions]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        rephrased_captions = []
        for output in outputs:
            try:
                generated_text = output.outputs[0].text
                rephrased_caption = process_llm_output(generated_text)
                rephrased_captions.append(rephrased_caption)
            except Exception as e:
                print(f"Error processing output: {e}")
        
        rephrased_captions_lists.append(rephrased_captions)
    
    end_time = time.time()
    print(f"Generation time (for {len(captions_lists)} entries): {end_time - start_time} seconds")
    
    df.loc[args.index_start:args.index_end - 1, "captions"] = pd.Series(rephrased_captions_lists).apply(str).values

def process_mcq_task(df, model_name, args, mcq_examples):
    """
    Processes the 'mcq' task by paraphrasing individual captions in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        model_name (str): The full model name to be used for the LLM.
        args (argparse.Namespace): Parsed command-line arguments.
        mcq_examples (list): Examples to be used in generating prompts for MCQ task.
        use_affirmation_negation_guideline (bool): Flag indicating whether to include the affirmation/negation guideline.
    """
    llm, sampling_params = initialize_llm(model_name)
    
    start_time = time.time()
    
    for i in tqdm(range(args.index_start, args.index_end), desc="Paraphrasing captions"):
        for j in range(4):  # Assuming there are 4 captions: "caption_0", "caption_1", "caption_2", "caption_3"
            caption = df.loc[i, f"caption_{j}"]
            prompt = generate_prompt(caption, mcq_examples, use_affirmation_negation_guideline=False)
            output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
            
            try:
                generated_text = output.outputs[0].text
                rephrased_caption = process_llm_output(generated_text)
                df.loc[i, f"caption_{j}"] = rephrased_caption
            except Exception as e:
                print(f"Error processing output for index {i}, caption_{j}: {e}")
    
    end_time = time.time()
    print(f"Generation time (for {args.index_end - args.index_start} entries): {end_time - start_time} seconds")

def main(args):
    # Map model aliases to full model names
    model_mapping = {
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama3.1": "meta-llama/Meta-Llama-3.1-70B-Instruct"
    }
    
    model_name = model_mapping[args.model]  # Use alias to get full model name

    df = pd.read_csv(args.input_file)

    if args.index_end is None or args.index_end > len(df):
        args.index_end = len(df)  # Set index_end to the number of entries

    if args.task == "retrieval":
        process_retrieval_task(df, model_name, args, retrieval_examples, use_affirmation_negation_guideline=args.use_affirmation_negation_guideline)
    elif args.task == "mcq":
        process_mcq_task(df, model_name, args, mcq_examples)

    if args.index_start == 0 and args.index_end == len(df):
        output_file_name = f"{args.input_file[:-4]}_{args.model}_rephrased.csv"
    else:
        output_file_name = f"{args.input_file[:-4]}_{args.model}_rephrased_{args.index_start}_{args.index_end}.csv"
    df.to_csv(output_file_name, index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)