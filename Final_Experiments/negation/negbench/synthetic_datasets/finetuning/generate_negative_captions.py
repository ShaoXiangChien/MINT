"""
generate_negative_captions.py: Generates negative captions from affirmative captions using a large language model (LLM).

### Functionality:
- Converts affirmative captions into negative captions by incorporating negation statements for objects absent in the image.
- Supports diverse caption structures with affirmation-first or negation-first styles.
- Randomly samples negative objects to create captions, ensuring diversity.

### Example Usage:
```bash
python generate_negative_captions.py --model llama3.1 --input_file path/to/input.csv --output_base path/to/output --index_start 0 --index_end 100
```

### Input:
- A CSV file with columns:
  - `caption`: Affirmative captions describing the image.
  - `negative_objects`: List of objects absent in the image.

### Output:
- A CSV file with a new column:
  - `negative_captions`: List of generated negative captions.

### Notes:
- Uses pre-defined examples for affirmation-first and negation-first styles.
- Median (or fixed) number of negative objects (`M`) is used for sampling.
- Supports models: `mixtral` and `llama3.1`.
"""

import argparse
import pandas as pd
from vllm import LLM, SamplingParams
import time
from tqdm import tqdm
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate negative captions from affirmative captions using a large language model.")
    parser.add_argument("--input_file", type=str, required=True, help="The input CSV file containing captions and object lists.")
    parser.add_argument("--index_start", type=int, default=0, help="The starting index of rows to process (inclusive).")
    parser.add_argument("--index_end", type=int, default=-1, help="The ending index of rows to process (exclusive).")
    parser.add_argument("--output_base", type=str, default="output", help="The base name for the output CSV file.")
    parser.add_argument("--model", type=str, default="mixtral", choices=["mixtral", "llama3.1"], 
                        help="The LLM model to use for generating negative captions. Options: 'mixtral' or 'llama3.1'.")
    return parser.parse_args()

affirmation_first_examples_list = [
    """Affirmative Caption: A beautiful park with lush trees and a fountain.
Negative Object: [Playground]
Negative Caption: A beautiful park with lush trees and a fountain, but no playground is present.""",

    """Affirmative Caption: A cozy living room with a fireplace and a comfortable couch.
Negative Object: [TV]
Negative Caption: A cozy living room with a fireplace and a comfortable couch, but there is no TV.""",

    """Affirmative Caption: A bustling city street with tall buildings and busy traffic.
Negative Object: [Park]
Negative Caption: The street is busy with tall buildings and traffic, but lacks a park.""",

    """Affirmative Caption: A peaceful beach with golden sand and clear blue water.
Negative Object: [Boat]
Negative Caption: A peaceful beach with golden sand and clear blue water, though there isn’t a boat in sight.""",

    """Affirmative Caption: A vibrant fruit market filled with apples, oranges, and bananas.
Negative Object: [Vegetables]
Negative Caption: The market is filled with apples, oranges, and bananas, with no vegetables to be found.""",

    """Affirmative Caption: A kitchen with modern appliances and a spacious countertop.
Negative Object: [Dishwasher]
Negative Caption: A kitchen with modern appliances and a spacious countertop, lacking a dishwasher.""",

    """Affirmative Caption: A cozy bedroom with a large bed and soft lighting.
Negative Object: [Bookshelf]
Negative Caption: A cozy bedroom with a large bed and soft lighting, though it has no bookshelf.""",

    """Affirmative Caption: A busy café with people enjoying their coffee and pastries.
Negative Object: [Outdoor seating]
Negative Caption: A busy café where people enjoy their coffee and pastries, but without any outdoor seating available.""",

    """Affirmative Caption: A serene mountain landscape with snow-covered peaks and clear skies.
Negative Object: [Lake]
Negative Caption: Snow-covered peaks and clear skies define the landscape, yet there’s no lake in view.""",

    """Affirmative Caption: A modern office space with desks and computers.
Negative Object: [Conference room]
Negative Caption: A modern office with desks and computers, though it is missing a conference room.""",

    """Affirmative Caption: A charming garden with colorful flowers and a wooden bench.
Negative Object: [Fountain]
Negative Caption: A charming garden with colorful flowers and a wooden bench, though no fountain is present.""",

    """Affirmative Caption: A busy airport terminal filled with travelers and suitcases.
Negative Object: [Flight attendants]
Negative Caption: The airport terminal is bustling with travelers and suitcases, yet there are no flight attendants around.""",

    """Affirmative Caption: A large stadium filled with cheering fans and bright lights.
Negative Object: [Scoreboard]
Negative Caption: The stadium is alive with cheering fans and bright lights, but there’s no scoreboard visible.""",

    """Affirmative Caption: A quiet library with rows of bookshelves and reading tables.
Negative Object: [Computers]
Negative Caption: A quiet library with rows of bookshelves and reading tables, though no computers can be found.""",

    """Affirmative Caption: A sunny playground with children playing on swings and slides.
Negative Object: [Picnic table]
Negative Caption: The playground is sunny, with children enjoying swings and slides, but it has no picnic table.""",

    """Affirmative Caption: A stylish café with modern décor and cozy chairs.
Negative Object: [Artwork]
Negative Caption: A stylish café with modern décor and cozy chairs, though there’s no artwork on display.""",

    """Affirmative Caption: A picturesque mountain cabin surrounded by tall trees.
Negative Object: [Lake]
Negative Caption: A picturesque mountain cabin surrounded by tall trees, yet no lake can be found nearby.""",

    """Affirmative Caption: A modern gym with treadmills, weights, and mirrors.
Negative Object: [Swimming pool]
Negative Caption: The gym is equipped with treadmills, weights, and mirrors, but it doesn’t have a swimming pool.""",

    """Affirmative Caption: A busy kitchen with chefs preparing dishes and fresh ingredients.
Negative Object: [Oven]
Negative Caption: The kitchen is full of chefs preparing fresh ingredients, though an oven is absent.""",

    """Affirmative Caption: A peaceful park with walking paths and tall trees.
Negative Object: [Fountain]
Negative Caption: A peaceful park with walking paths and tall trees, but no fountain in sight.""",
]

negation_first_examples_list = [
    """Affirmative Caption: A vibrant garden filled with blooming flowers and fluttering butterflies.
Negative Object: [Trees]
Negative Caption: No trees are present in this vibrant garden, but it is filled with blooming flowers and fluttering butterflies.""",

    """Affirmative Caption: A serene lake surrounded by mountains and under a clear blue sky.
Negative Object: [Boats]
Negative Caption: There are no boats on the serene lake, but the surrounding mountains and clear blue sky create a breathtaking view.""",

    """Affirmative Caption: A bustling city street lined with shops and filled with people walking.
Negative Object: [Cars]
Negative Caption: While there are no cars on this bustling city street, it is lined with shops and filled with people walking.""",

    """Affirmative Caption: A cozy living room with soft lighting and comfortable seating.
Negative Object: [TV]
Negative Caption: No TV can be found in the cozy living room, though it has soft lighting and comfortable seating.""",

    """Affirmative Caption: A clean kitchen with a large sink and modern appliances.
Negative Object: [Microwave]
Negative Caption: There is no microwave in this clean kitchen, but it does have a large sink and modern appliances.""",

    """Affirmative Caption: A scenic view of a beach with golden sand and gentle waves.
Negative Object: [Umbrellas]
Negative Caption: No umbrellas dot the golden sandy beach, yet the gentle waves and scenic view remain inviting.""",

    """Affirmative Caption: A quaint café with wooden tables and a variety of pastries on display.
Negative Object: [Customers]
Negative Caption: The café is devoid of customers, though it features wooden tables and an assortment of pastries on display.""",

    """Affirmative Caption: A beautiful garden with vibrant flowers and a stone pathway.
Negative Object: [Fountain]
Negative Caption: There is no fountain in this garden, but the vibrant flowers and stone pathway add plenty of charm.""",

    """Affirmative Caption: A bright and airy office with large windows and plants.
Negative Object: [Computers]
Negative Caption: No computers are visible in the bright, airy office, though it’s filled with plants and has large windows.""",

    """Affirmative Caption: A peaceful park with a large open lawn and walking paths.
Negative Object: [Playground]
Negative Caption: There is no playground in the peaceful park, but the large open lawn and walking paths are inviting.""",

    """Affirmative Caption: A bustling market filled with fresh produce and colorful stalls.
Negative Object: [Meat]
Negative Caption: No meat is sold at this bustling market, yet it is packed with fresh produce and colorful stalls.""",

    """Affirmative Caption: A beautiful sunset over the ocean with waves gently lapping the shore.
Negative Object: [People]
Negative Caption: There are no people on the beach at sunset, but the ocean waves gently lap the shore in a serene setting.""",

    """Affirmative Caption: A quiet street lined with trees and charming houses.
Negative Object: [Cars]
Negative Caption: No cars are parked along the quiet street, though the charming houses and tree-lined sidewalks give it life.""",

    """Affirmative Caption: A well-organized desk with books and a lamp.
Negative Object: [Computer]
Negative Caption: No computer can be seen on the organized desk, but it’s neatly arranged with books and a lamp.""",

    """Affirmative Caption: A lively park filled with children playing and a fountain at its center.
Negative Object: [Benches]
Negative Caption: No benches are available in the lively park, but children are playing near the central fountain.""",

    """Affirmative Caption: A peaceful lake surrounded by tall grass and a clear sky.
Negative Object: [Birds]
Negative Caption: There are no birds flying around the peaceful lake, but the tall grass and clear sky offer a serene view.""",

    """Affirmative Caption: A bright and spacious kitchen with a marble countertop and modern stove.
Negative Object: [Fridge]
Negative Caption: There is no fridge in this spacious kitchen, though it boasts a marble countertop and a modern stove.""",

    """Affirmative Caption: A rustic cabin with a wooden porch and a view of the mountains.
Negative Object: [Chimney]
Negative Caption: The cabin lacks a chimney, but its wooden porch offers a stunning view of the mountains.""",

    """Affirmative Caption: A calm beach with gentle waves and soft sand.
Negative Object: [Surfboards]
Negative Caption: No surfboards are visible on the calm beach, though the gentle waves and soft sand are perfect for relaxation.""",

    """Affirmative Caption: A quiet library filled with bookshelves and reading tables.
Negative Object: [Computers]
Negative Caption: No computers are available in the quiet library, but there are plenty of bookshelves and reading tables.""",

    """Affirmative Caption: A modern living room with large windows and stylish furniture.
Negative Object: [TV]
Negative Caption: The living room has no TV, but the large windows and stylish furniture make it an inviting space."""
]

def compute_median_negative_objects(df):
    """
    Computes the median number of negative objects per image across the entire DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data, including a column for negative objects.

    Returns:
        int: The median number of negative objects per image.
    """
    negative_object_counts = df["negative_objects"].apply(lambda x: len(eval(x)))
    median_count = int(np.median(negative_object_counts))
    return median_count

def sample_negative_objects(negative_objects, M):
    """
    Samples M negative objects from the list of negative objects.
    If the list has fewer than M objects, sampling is done with replacement.
    Otherwise, sampling is done without replacement.

    Args:
        negative_objects (list): The list of negative objects.
        M (int): The number of objects to sample.

    Returns:
        list: A list of M sampled negative objects.
    """
    if len(negative_objects) < M:
        return random.choices(negative_objects, k=M)
    else:
        return random.sample(negative_objects, k=M)

def generate_prompt(affirmative_caption, negative_objects, examples):
    """
    Generates a prompt for an LLM to create a negative caption based on an affirmative caption, 
    a list of positive objects present in the image, and a list of negative objects absent from the image.

    Args:
        affirmative_caption (str): The affirmative caption describing the image.
        positive_objects (list): A list of objects that are present in the image.
        negative_objects (list): A list of objects that are absent from the image.
        examples (str): Example prompts and responses to guide the LLM's output.

    Returns:
        str: A formatted prompt string to be used as input to the LLM.

    Example usage:
        >>> examples = '''Affirmative Caption: "A man is sitting on a bench with a dog."
        Negative Object: ["cat"]
        Negative Caption: "A man is sitting on a bench with a dog, but there is no cat anywhere in the scene."'''

        >>> generate_prompt("A cat is lying on a mat.", ["cat", "mat"], ["dog", "ball"], examples)
    """
    # Create the prompt template
    prompt_template = f"""You are given an affirmative caption describing an image. In addition, you will be provided a negative object that is absent from the image. Your task is to generate a negative caption, which includes negation statements specifying the absence of the negative object.

Instructions:

1. Affirm the presence of positive objects and negate the negative objects.
2. Do not introduce new objects.
3. Keep the captions concise, clear, and engaging, with diverse structures.
4. Follow the provided examples. If the example captions start with affirmation, start with affirmation. If the example captions start with negation, start with negation.

Examples:

{examples}

Affirmative Caption: "{affirmative_caption}"
Negative Object: [{', '.join(negative_objects)}]
Negative Caption: """
    
    return prompt_template

def process_llm_output(output_text):
    """
    Processes the LLM output to extract the generated negative caption.

    The function accounts for different formats the LLM might use, such as:
    - Directly outputting the negative caption.
    - Enclosing the negative caption in quotation marks.
    - Preceding the caption with "Negative Caption: ".
    - Outputting the caption with additional leading or trailing spaces.
    - Including additional lines with irrelevant content after the caption.

    Args:
        output_text (str): The text output by the LLM.

    Returns:
        str: The cleaned and extracted negative caption.

    Example usage:
        >>> process_llm_output('Negative Caption: "There is no car in the image, only a tree and a bench."\nOther content here.')
        'There is no car in the image, only a tree and a bench.'

        >>> process_llm_output('"A man stands by a tree, but no dog is present."\nAdditional info.')
        'A man stands by a tree, but no dog is present.'

        >>> process_llm_output('There is no bicycle or cat in the image.')
        'There is no bicycle or cat in the image.'

        >>> process_llm_output('Negative Caption: No cars are visible, but the image does show a tree.\nExplanation follows.')
        'No cars are visible, but the image does show a tree.'
    """
    # Trim any leading or trailing whitespace
    output_text = output_text.strip()

    # Extract the first line, assuming it contains the relevant caption
    output_text = output_text.split('\n', 1)[0].strip()
    
    # If the output starts with 'Negative Caption:', remove this prefix
    if output_text.startswith('Negative Caption:'):
        output_text = output_text[len('Negative Caption:'):].strip()

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
    try:
        print(f"Trying to initialize LLM with tensor_parallel_size=4...")
        llm = LLM(model=model_name, tensor_parallel_size=4)
    except Exception as e:
        print(f"Failed to initialize LLM with tensor_parallel_size=4: {e}")
        print(f"Initializing LLM with tensor_parallel_size=2...")
        llm = LLM(model=model_name, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=900, stop=["\n\n"])
    return llm, sampling_params

def process_captioning_task(df, model_name, args, M):
    """
    Processes the task by generating M negative captions for each affirmative caption in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data, including columns for captions, positive objects, and negative objects.
        model_name (str): The full model name to be used for the LLM.
        args (argparse.Namespace): Parsed command-line arguments.
        M (int): The median number of negative objects per image, used to determine the number of captions to generate.
    """
    # Initialize a new column to store the generated negative captions
    df["negative_captions"] = None

    prompts = []  # Collect all prompts to batch together
    image_indices = []  # Track the image index for each prompt
    object_indices = []  # Track the negative object index for each prompt (in case there are multiple negative objects per image)

    # Start generating prompts for all the captions
    for i in tqdm(df.index, desc="Generating prompts"):
        # Extract the necessary information from the DataFrame
        affirmative_caption = df.loc[i, "caption"]
        negative_objects = eval(df.loc[i, "negative_objects"])  # Assuming objects are stored as strings

        # Sample M negative objects from the list of negative objects
        sampled_negative_objects = sample_negative_objects(negative_objects, M)

        for j, negative_object in enumerate(sampled_negative_objects):
            # Generate a prompt for the LLM based on the affirmative caption, negative object, and examples
            if random.random() < 0.5:
                examples_list = affirmation_first_examples_list
            else:
                examples_list = negation_first_examples_list

            # Randomly select and shuffle three examples
            selected_examples = random.sample(examples_list, 3)
            random.shuffle(selected_examples)
            examples = "\n\n".join(selected_examples)

            prompt = generate_prompt(affirmative_caption, [negative_object], examples)

            # Add the prompt to the list
            prompts.append(prompt)
            image_indices.append(i)  # Track which image this prompt corresponds to
            object_indices.append(j)  # Track which negative object this prompt corresponds to

    # Initialize the LLM and sampling parameters
    llm, sampling_params = initialize_llm(model_name)

    # Send all prompts in a single batch to the LLM
    start_time = time.time()
    print("Generating negative captions...")
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    print(f"Generation time (for {len(prompts)} prompts): {end_time - start_time} seconds")

    # Initialize a dictionary to store the negative captions for each image
    negative_captions_dict = {i: [] for i in df.index}

    # Process the LLM outputs and organize them by image index
    for idx, output in enumerate(outputs):
        try:
            generated_text = output.outputs[0].text
            negative_caption = process_llm_output(generated_text)

            # Get the corresponding image and object index
            image_index = image_indices[idx]
            object_index = object_indices[idx]

            # Append the generated caption to the relevant image's list
            if negative_caption.strip():
                negative_captions_dict[image_index].append(negative_caption)

        except Exception as e:
            print(f"Error processing output for image index {image_index}, object index {object_index}: {e}")

    # Store the generated negative captions back into the DataFrame
    for i in df.index:
        negative_captions = negative_captions_dict[i]
        df.loc[i, "negative_captions"] = str(negative_captions)

def main(args):
    # Map model aliases to full model names
    model_mapping = {
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama3.1": "meta-llama/Meta-Llama-3.1-70B-Instruct"
    }
    
    model_name = model_mapping[args.model]  # Use alias to get full model name

    df = pd.read_csv(args.input_file)
    # M = compute_median_negative_objects(df)
    # print(f"Median number of negative objects per image: {M}")
    M = 3 # Set M to a fixed value for consistent results, or use the computed median
    print(f"Using M = {M} for generating negative captions.")

    # Set index_end to the length of the DataFrame if it's -1 or exceeds the length of the DataFrame
    if args.index_end == -1 or args.index_end > len(df):
        args.index_end = len(df)

    # Subset the DataFrame to only the rows specified by index_start and index_end
    df_subset = df.iloc[args.index_start:args.index_end].copy()

    # Store the initial number of rows before filtering
    initial_rows = len(df_subset)

    # Remove rows where the negative objects are empty
    df_subset = df_subset[df_subset["negative_objects"].apply(lambda x: bool(eval(x)) if pd.notnull(x) else False)].copy()

    # Calculate and print the number of dropped rows
    dropped_rows = initial_rows - len(df_subset)
    print(f"Dropped {dropped_rows} rows where negative objects were empty.")

    # Process the task of generating negative captions
    process_captioning_task(df_subset, model_name, args, M)

    # Generate output file name based on the processed range
    if args.index_start == 0 and args.index_end == len(df):
        output_file_name = f"{args.output_base}_{args.model}_neg_captions.csv"
    else:
        output_file_name = f"{args.output_base}_{args.model}_neg_captions_{args.index_start}_{args.index_end}.csv"

    # Save the modified DataFrame with the new negative captions
    df_subset.to_csv(output_file_name, index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)
