"""
This script performs various tasks for generating and processing video datasets for negation evaluation using a Large Language Model (LLM). 
It includes three main tasks:
1. **Concepts**: Extract two positive concepts and one negative concept for each video, and generate negated captions for retrieval tasks.
2. **Retrieval**: Paraphrase the negated captions to create linguistically diverse retrieval datasets.
3. **MCQ**: Paraphrase captions for multiple-choice questions (MCQs) to ensure linguistic diversity in MCQ datasets.

The script processes input CSV files and produces outputs based on the selected task.

---

### Input CSV File Format
The input CSV file must include:
- `image_id`: Unique identifier for each video.
- `all_captions`: A list of original captions describing the video (in string format).
- `filepath`: Path to the video file.
- For retrieval or MCQ tasks:
  - `captions`: A list of retrieval captions (in string format) for the retrieval task.
  - `caption_0`, `caption_1`, `caption_2`, `caption_3`: MCQ captions for the MCQ task.

---

### Output CSV File Format
The output format depends on the selected task:
1. **Concepts**:
   - Adds columns:
     - `positive_concept1`: The first positive concept.
     - `positive_concept2`: The second positive concept.
     - `negative_concept`: The negative concept.
     - `captions`: Updated list of captions with negated concepts.
   - Columns are reordered as: `image_id`, `captions`, `positive_concept1`, `positive_concept2`, `negative_concept`, `all_captions`, `filepath`.

2. **Retrieval**:
   - Updates the `captions` column with paraphrased retrieval captions.

3. **MCQ**:
   - Updates `caption_0`, `caption_1`, `caption_2`, `caption_3` with paraphrased captions.

---

### Example Usage

1. **Concepts Task**: Extract positive and negative concepts for MSRVTT dataset.
```bash
python process_video_tasks.py --input_file data/video/MSRVTT/msr_vtt_retrieval.csv \
    --output_base data/video/MSRVTT/negation/msr_vtt_retrieval_templated \
    --task concepts \
    --model mixtral
```

2. **Retrieval Task**: Paraphrase negated retrieval captions.
```bash
python process_video_tasks.py --input_file data/video/MSRVTT/negation/msr_vtt_retrieval_templated.csv \
    --output_base data/video/MSRVTT/negation/msr_vtt_retrieval_rephrased \
    --task retrieval \
    --use_affirmation_negation_guideline
```

3. **MCQ Task**: Paraphrase multiple-choice question captions.
```bash
python process_video_tasks.py --input_file data/video/MSRVTT/negation/msr_vtt_mcq_templated.csv \
    --output_base data/video/MSRVTT/negation/msr_vtt_mcq_rephrased \
    --task mcq \
    --model llama3.1
```

---

### Notes
- The LLM model can be set to `mixtral` or `llama3.1` using the `--model` argument.
- Use the `--use_affirmation_negation_guideline` flag to ensure affirmation/negation order is preserved during paraphrasing.
- If processing a large dataset, you can use the `--index_start` and `--index_end` arguments to process rows in chunks.
"""

import argparse
import pandas as pd
import random
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Generate negative captions for video tasks using a large language model.")
    parser.add_argument("--input_file", type=str, required=True, help="The input CSV file containing captions and relevant metadata.")
    parser.add_argument("--index_start", type=int, default=0, help="The starting index of rows to process (inclusive).")
    parser.add_argument("--index_end", type=int, default=-1, help="The ending index of rows to process (exclusive).")
    parser.add_argument("--output_base", type=str, default="output", help="The base name for the output CSV file.")
    parser.add_argument("--model", type=str, default="mixtral", choices=["mixtral", "llama3.1"], 
                        help="The LLM model to use for generating negative captions. Options: 'mixtral' or 'llama3.1'.")
    parser.add_argument("--task", type=str, default="concepts", choices=["concepts", "retrieval", "mcq"],
                        help="The type of task to perform. Options: 'concepts', 'retrieval', 'mcq'.")
    parser.add_argument('--use_affirmation_negation_guideline', action='store_true', 
                        help="Include affirmation/negation guideline in paraphrasing task.") # if passed, it is true, else false
    return parser.parse_args()

# The following examples might not be needed anymore # TODO: Check if these examples are needed
affirmation_first_examples_list = [
    """Original Captions: ['a cartoon clip of pokemon dancing', 'a group of pokemon dance on barrels while a treecko and mudkip dance together', 'a scene from a pokemon movie', 'cartoon characters are dancing', 'pokemon characters stand on top of barrels', 'pokemon dance on barrels together']
Negative Concept: Fight
Negative Caption: The pokemon are joyfully dancing together, not engaging in any fights.""",

    """Original Captions: ['a chef is chopping vegetables in a kitchen', 'a person is preparing a meal', 'a chef working in a busy kitchen', 'a cook carefully prepares ingredients', 'vegetables being chopped for a meal']
Negative Concept: Fire
Negative Caption: In the kitchen, the chef focuses on chopping vegetables, with no fire involved in the process.""",

    """Original Captions: ['a person is riding a bicycle down a quiet street', 'a cyclist is pedaling along a tree-lined road', 'a cyclist is cruising on an empty road', 'a person on a bicycle rides through a park', 'a peaceful bike ride on a sunny day', 'someone is riding a bike on a tranquil path', 'a cyclist is pedaling through a quiet neighborhood']
Negative Concept: Traffic
Negative Caption: The cyclist pedals leisurely through a serene area, encountering no traffic along the way."""
]
negation_first_examples_list = [
    """Original Captions: ['a dog is playing fetch with its owner in a large park', 'a person throws a ball for the dog to retrieve', 'a dog catching a ball in mid-air', 'a dog is joyfully playing with a ball', 'a person throws a ball, and the dog runs after it']
Negative Concept: Barking
Negative Caption: Not barking loudly, the dog joyfully fetches the ball thrown by its owner.""",

    """Original Captions: ['a group of people are gathered around a campfire at night', 'friends are roasting marshmallows over the fire', 'people are sitting on logs around the fire', 'a campfire is surrounded by people enjoying the night']
Negative Concept: Rain
Negative Caption: There is no rain falling, just friends enjoying the warmth of the campfire at night.""",

    """Original Captions: ['a woman is painting a landscape on a canvas', 'an artist is focused on her painting', 'a painter carefully adds details to the canvas', 'a person is creating a landscape painting', 'a person paints outdoors, surrounded by nature']
Negative Concept: Mess
Negative Caption: Without creating any mess, the artist carefully paints a landscape on her canvas."""
]

# These examples are used for the 'concepts' task
concepts_examples_list = [
    """original captions: ['a cartoon clip of pokemon dancing', 'a group of pokemon dance on barrels while a treecko and mudkip dance together', 'a scene from a pokemon movie', 'cartoon characters are dancing', 'pokemon characters stand on top of barrels', 'pokemon dance on barrels together']
positive1: pokemon
positive2: dancing
negative: fighting""",

    """original captions: ['a chef is chopping vegetables in a kitchen', 'a person is preparing a meal', 'a cook is slicing ingredients on a cutting board', 'someone is dicing vegetables for a dish', 'a chef working in a busy kitchen', 'a cook carefully prepares ingredients', 'vegetables being chopped for a meal', 'a chef’s hands are chopping fresh vegetables', 'a cook preparing a meal in a professional kitchen', 'ingredients being prepared for cooking']
positive1: chef
positive2: chopping vegetables
negative: baking""",

    """original captions: ['a soccer player dribbles the ball', 'a player is running on the field with a ball', 'a soccer match is in progress', 'an athlete skillfully moves the ball down the field', 'a player kicks the ball towards the goal', 'a soccer game being played on a grassy field']
positive1: soccer
positive2: dribbling
negative: swimming""",

    """original captions: ['a musician is playing the guitar', 'a person strumming chords on a guitar', 'a guitarist performing on stage', 'someone playing an acoustic guitar', 'a live performance featuring guitar music', 'a close-up of a guitarist’s hands while playing']
positive1: guitar
positive2: playing
negative: singing""",

    """original captions: ['a child is blowing bubbles', 'a kid is playing with a bubble wand', 'a child creating bubbles in the air', 'a young boy blowing soap bubbles', 'bubbles floating around as a child plays', 'a child joyfully makes bubbles in the park']
positive1: child
positive2: bubbles
negative: kite""",

    """original captions: ['a dog is fetching a stick', 'a dog runs after a thrown stick', 'a dog is playing fetch in the park', 'a playful dog retrieves a stick', 'a dog catches a stick in mid-air', 'a dog running with a stick in its mouth']
positive1: dog
positive2: stick
negative: ball""",

    """original captions: ['a woman is painting on a canvas', 'an artist is creating a colorful painting', 'a person is using brushes to paint', 'a painter is working on a new piece of art', 'a woman carefully applying paint to a canvas', 'an artist mixing colors on a palette']
positive1: painting
positive2: canvas
negative: drawing""",

    """original captions: ['a person is skiing down a snowy hill', 'a skier is racing down a slope', 'someone skiing in the mountains', 'a skier gliding through fresh snow', 'a person skiing at high speed', 'a skier navigating a steep slope']
positive1: skiing
positive2: snow
negative: surfing""",

    """original captions: ['a cat is climbing a tree', 'a cat is perched on a branch', 'a cat is scaling a tall tree', 'a feline is climbing up a tree', 'a cat climbs higher on a tree branch', 'a cat making its way up a tree']
positive1: cat
positive2: climbing
negative: flying""",

    """original captions: ['a person is typing on a laptop', 'someone is working on a computer', 'a person is typing an email', 'a close-up of hands typing on a keyboard', 'a person working on a laptop in an office', 'someone is writing code on a computer']
positive1: typing
positive2: laptop
negative: reading""",

    """original captions: ['a child is riding a bicycle', 'a young girl is cycling down the street', 'a kid riding a bike in the park', 'a child pedaling a bicycle', 'a boy riding a bike with training wheels', 'a child enjoying a bike ride on a sunny day']
positive1: child
positive2: bicycle
negative: skateboard""",

    """original captions: ['a person is pouring coffee into a mug', 'a barista is making a cup of coffee', 'a steaming cup of coffee being prepared', 'someone is pouring hot coffee into a cup', 'a close-up of coffee being poured into a mug', 'a freshly brewed cup of coffee']
positive1: coffee
positive2: pouring
negative: tea""",

    """original captions: ['a group of friends are hiking on a trail', 'hikers walking through a forest', 'a group of people climbing a mountain', 'a person hiking up a steep hill', 'a group of hikers exploring the wilderness', 'friends hiking together in nature']
positive1: hiking
positive2: trail
negative: swimming""",

    """original captions: ['a person is playing the piano', 'a pianist is performing a classical piece', 'a close-up of hands playing piano keys', 'someone is playing a grand piano', 'a pianist is performing on stage', 'a person playing a piano in a concert']
positive1: piano
positive2: playing
negative: violin""",

    """original captions: ['a person is baking a cake in the kitchen', 'a cake is being mixed and prepared', 'someone is decorating a cake with frosting', 'a cake is being baked in the oven', 'a person adding ingredients to a cake mix', 'a freshly baked cake being taken out of the oven']
positive1: cake
positive2: baking
negative: frying"""
]

# These examples are used for the 'paraphrase_mcq' task
paraphrasing_mcq_examples_list = [
    """Original Caption: This video features pokemon and dancing
Rephrased Caption: Pokemon and dancing are both featured in this video.""",

    """Original Caption: This video does not feature baking
Rephrased Caption: Baking is not included in the content.""",

    """Original Caption: This video features fighting, but not dancing
Rephrased Caption: This video includes fighting, but no dancing.""",

    """Original Caption: This video features baking
Rephrased Caption: Baking takes the spotlight here.""",

    """Original Caption: This video features chopping vegetables, but not chef
Rephrased Caption: Vegetable chopping is shown, but no chef appears.""",

    """Original Caption: This video features soccer dribbling, but no swimming
Rephrased Caption: This video focuses on soccer dribbling, with swimming absent.""",

    """Original Caption: This video features singing, but no guitar
Rephrased Caption: Singing is present, yet no guitar is involved.""",

    """Original Caption: This video features bubbles
Rephrased Caption: Bubbles are prominently featured.""",

    """Original Caption: This video does not feature dog
Rephrased Caption: A dog does not make an appearance.""",

    """Original Caption: This video does not feature drawing
Rephrased Caption: Drawing is absent from the content.""",

    """Original Caption: This video features painting
Rephrased Caption: This video highlights painting.""",

    """Original Caption: This video features skiing, but no surfing
Rephrased Caption: Skiing is showcased, while surfing is not.""",

    """Original Caption: This video features surfing, but no skiing
Rephrased Caption: Surfing takes center stage, with skiing left out.""",

    """Original Caption: This video features coffee, but no pouring
Rephrased Caption: Coffee is featured, but pouring is not.""",

    """Original Caption: This video features swimming
Rephrased Caption: Swimming is displayed in this content.""",

    """Original Caption: This video features a cat, but not a dog
Rephrased Caption: A cat appears, but there is no dog in this video.""",

    """Original Caption: This video features running, but no walking
Rephrased Caption: Running is emphasized here, while walking is excluded from this video.""",

    """Original Caption: This video does not feature cooking
Rephrased Caption: Cooking is missing from the video.""",

    """Original Caption: This video features piano playing
Rephrased Caption: Piano playing is showcased.""",

    """Original Caption: This video does not feature birds
Rephrased Caption: Birds are not included in the scenes."""
]

# These examples are used for the 'paraphrase_retrieval' task
paraphrasing_retrieval_examples_list = [
    """Original Caption: There is no rain in the video. A man is walking through a sunny park.
Rephrased Caption: No rain is falling in the video, just a man walking through a sunny park.""",

    """Original Caption: There is no singing in the video. A group of friends is having a silent picnic.
Rephrased Caption: No singing is heard as a group of friends enjoys a quiet picnic.""",

    """Original Caption: There is no explosion in the video. A car is slowly driving down a quiet road.
Rephrased Caption: No explosion occurs in the video, only a car slowly driving down a quiet road.""",

    """Original Caption: There is no swimming in the video. A family is building sandcastles on the beach.
Rephrased Caption: No one is swimming in the video, just a family building sandcastles on the beach.""",

    """Original Caption: There is no snow in the video. A child is playing outside under a clear blue sky.
Rephrased Caption: No snow appears in the video, with a child playing outside under a clear blue sky.""",

    """Original Caption: A cat is chasing a toy mouse. There is no barking in the video.
Rephrased Caption: A cat chases a toy mouse, and there’s no barking to interrupt the scene.""",

    """Original Caption: A chef is preparing a gourmet meal. There is no talking in the video.
Rephrased Caption: A chef is preparing a gourmet meal, with the kitchen quiet and no talking heard.""",

    """Original Caption: Two people are dancing in a ballroom. There is no clapping in the video.
Rephrased Caption: Two people are dancing elegantly in a ballroom, and the scene is silent with no clapping.""",

    """Original Caption: A bird is flying over the ocean. There is no ship in the video.
Rephrased Caption: A bird flies over the vast ocean, with no ships in sight.""",

    """Original Caption: Children are playing in a park. There is no music in the video.
Rephrased Caption: Children play energetically in a park, and there’s no music playing in the background.""",

    """Original Caption: There is no running in the video. A woman is standing still on a hill.
Rephrased Caption: No one is running in the video; instead, a woman is calmly standing on a hill.""",

    """Original Caption: There is no sound in the video. A dog is silently chasing its tail.
Rephrased Caption: No sound accompanies the video, where a dog is silently chasing its tail.""",

    """Original Caption: There is no laughing in the video. Two kids are quietly playing with toys.
Rephrased Caption: No laughter can be heard in the video, as two kids quietly play with their toys.""",

    """Original Caption: There is no train in the video. A man is waiting alone at a station.
Rephrased Caption: No train is present in the video; a man waits alone at the station.""",

    """Original Caption: There is no cooking in the video. A chef is carefully arranging dishes on a table.
Rephrased Caption: No cooking takes place in the video, only a chef carefully arranging dishes on a table.""",

    """Original Caption: A bird is perched on a branch. There is no wind in the video.
Rephrased Caption: A bird perches on a branch, and the scene is calm with no wind.""",

    """Original Caption: A musician is tuning a guitar. There is no singing in the video.
Rephrased Caption: A musician is tuning a guitar, with no singing accompanying the action.""",

    """Original Caption: A boat is floating on a lake. There is no one swimming in the video.
Rephrased Caption: A boat floats peacefully on a lake, with no one swimming nearby.""",

    """Original Caption: A cat is curled up on a couch. There is no one else in the video.
Rephrased Caption: A cat is curled up on a couch, alone in the video with no one else around.""",

    """Original Caption: A group of dancers is performing on stage. There is no audience in the video.
Rephrased Caption: A group of dancers performs on stage, and there’s no audience in the video."""
]

def initialize_llm(model_name):
    """
    Initializes the LLM model and sets up the sampling parameters.

    Parameters:
        model_name (str): The full model name to be used for the LLM.

    Returns:
        llm (LLM): The initialized LLM object.
        sampling_params (SamplingParams): The configured sampling parameters.
    """
    if "llama" in model_name:
        try:
            print(f"Trying to initialize LLM with tensor_parallel_size=4")
            llm = LLM(model=model_name, tensor_parallel_size=4)
        except:
            print(f"Failed to initialize LLM with tensor_parallel_size=4. Using default value (2).")
            llm = LLM(model=model_name, tensor_parallel_size=2)
    else:
        llm = LLM(model=model_name, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=900, stop=["\n\n"])
    return llm, sampling_params

# TODO: determine if needed, or just use paraphrasing task
def generate_prompt_retrieval(original_captions, examples):
    """
    Generates a prompt for an LLM to create a negative caption based on a list of original captions 
    describing a video. The LLM will also identify the related negative concept.

    Args:
        original_captions (list): A list of captions describing the video.
        examples (str): Example prompts and responses to guide the LLM's output.

    Returns:
        str: A formatted prompt string to be used as input to the LLM.
    """
    # Create the prompt template
    prompt_template = f"""Your task is to analyze a list of captions describing a video and identify a related negative concept. A negative concept is an idea, object, or action that could logically relate to the video but is absent from the provided captions. After identifying the negative concept, you will generate a negative caption that explicitly states the absence of this concept using negation language. Ensure the negative caption aligns with the video's content as described by the given captions.

Instructions:
1. The caption should include affirmations from the existing captions, while also negating the negative concept.
2. Keep the captions concise, clear, and engaging, with diverse structures.
3. Follow the provided examples. If the example captions start with affirmation, start with affirmation. If the example captions start with negation, start with negation. Do not include any additional words or explanations.

Examples:

{examples}

Original Captions: {original_captions}"""
    
    return prompt_template

def generate_prompt_concepts(original_captions, examples):
    """
    Generates a prompt for the 'concepts' task.

    Args:
        original_captions (list): A list of captions describing the video.
        examples (str): Example prompts and responses to guide the LLM's output.

    Returns:
        str: A formatted prompt string to be used as input to the LLM.
    """
    prompt_template = f"""Your task is to analyze a list of captions describing a video and identify two positive concepts and one negative concept. A positive concept is an object or action that describes the video visuals, and a negative concept is an object or action that could relate to the video but is absent from the provided captions. Do not include any additional words or explanations.

Examples:

{examples}

original captions: {original_captions}"""
    
    return prompt_template

def generate_prompt_paraphrasing(original_caption, examples, use_affirmation_negation_guideline):
    """
    Generates a prompt for the 'paraphrase' task.

    Args:
        original_caption (str): The original caption to be paraphrased.
        examples (str): Example prompts and responses to guide the LLM's output.
        use_affirmation_negation_guideline (bool): Flag indicating whether to include the affirmation/negation order guideline.

    Returns:
        str: A formatted prompt string to be used as input to the LLM.
    """
    guidelines = """
1. Do not introduce any new concepts.
2. Only output the rephrased caption without additional text or explanations.
"""
    
    if use_affirmation_negation_guideline:
        guidelines = """
1. Do not introduce any new concepts.
2. If the original caption starts with the affirmation, start with affirmation. If the original caption starts with negation, start with negation. In either way, make it flow more naturally.
3. Only output the rephrased caption without additional text or explanations.
"""

    prompt_template = f"""You will be given a video caption that describes the visual presence of certain concepts. A concept can be an action or object. The caption may also describe the absence of some concept, or both presence and absence. Your task is to rephrase the caption to improve its flow and make it more engaging. While rephrasing, please follow these guidelines:

{guidelines}

Examples:

{examples}

Original Caption: "{original_caption}"
Rephrased Caption: """
    
    return prompt_template

def generate_prompt(task, original_captions, examples):
    """
    Generates a prompt for an LLM based on the specified task.

    Args:
        task (str): The task type, e.g., 'retrieval', 'paraphrase', 'concepts'.
        original_captions (list): A list of captions describing the video.
        examples (str): Example prompts and responses to guide the LLM's output.

    Returns:
        str: A formatted prompt string to be used as input to the LLM.
    """
    if task == "retrieval":
        return generate_prompt_retrieval(original_captions, examples)
    elif task == "concepts":
        return generate_prompt_concepts(original_captions, examples)
    elif task == "paraphrase":
        return generate_prompt_paraphrasing(original_captions, examples)
    else:
        raise ValueError(f"Cannot generate prompt for unknown task: {task}")

def process_llm_output_caption(output_text):
    """
    Processes the LLM output to extract the generated negative concept and negative caption.

    The function expects the LLM output to contain two lines:
    - The first line is the negative concept.
    - The second line is the negative caption.

    This function is case-insensitive for the terms "Negative Concept" and "Negative Caption".

    Args:
        output_text (str): The text output by the LLM.

    Returns:
        tuple: A tuple containing the negative concept (str) and the negative caption (str).

    Example usage:
        >>> output_text = "negative concept: Traffic\nNegative caption: The cyclist pedals leisurely through a serene area, encountering no traffic along the way."
        >>> negative_concept, negative_caption = process_llm_output(output_text)
        >>> print(negative_concept)
        'Traffic'
        >>> print(negative_caption)
        'The cyclist pedals leisurely through a serene area, encountering no traffic along the way.'
    """
    # Split the output into lines
    lines = output_text.strip().split('\n')

    # Initialize variables to store the concept and caption
    negative_concept = ""
    negative_caption = ""

    # Process each line
    for line in lines:
        # Normalize the line to lowercase for matching purposes
        normalized_line = line.lower().strip()

        # Check if the line starts with "negative concept:" and extract the concept
        if normalized_line.startswith("negative concept:"):
            negative_concept = line[len("Negative Concept:"):].strip()
        
        # Check if the line starts with "negative caption:" and extract the caption
        elif normalized_line.startswith("negative caption:"):
            negative_caption = line[len("Negative Caption:"):].strip()

    return negative_concept, negative_caption

def process_llm_output_concepts(output_text):
    """
    Processes the LLM output to extract two positive concepts and one negative concept.

    The function expects the LLM output to contain three lines:
    - The first line is the first positive concept, starting with "positive1:".
    - The second line is the second positive concept, starting with "positive2:".
    - The third line is the negative concept, starting with "negative:".

    This function is case-insensitive for the terms "positive1:", "positive2:", and "negative:".

    Args:
        output_text (str): The text output by the LLM.

    Returns:
        tuple: A tuple containing two positive concepts (str) and one negative concept (str).

    Example usage:
        >>> output_text = "positive1: Happiness\npositive2: Friendship\nnegative: Conflict"
        >>> positive1, positive2, negative_concept = process_llm_output_concepts(output_text)
        >>> print(positive1)
        'Happiness'
        >>> print(positive2)
        'Friendship'
        >>> print(negative_concept)
        'Conflict'
    """
    # Split the output into lines
    lines = output_text.strip().split('\n')

    # Initialize variables to store the concepts
    positive_concept1 = ""
    positive_concept2 = ""
    negative_concept = ""

    # Process each line
    for line in lines:
        # Normalize the line to lowercase for matching purposes
        normalized_line = line.lower().strip()

        # Check if the line starts with "positive1:" and extract the first positive concept
        if normalized_line.startswith("positive1:"):
            positive_concept1 = line[len("positive1:"):].strip()
        
        # Check if the line starts with "positive2:" and extract the second positive concept
        elif normalized_line.startswith("positive2:"):
            positive_concept2 = line[len("positive2:"):].strip()

        # Check if the line starts with "negative:" and extract the negative concept
        elif normalized_line.startswith("negative:"):
            negative_concept = line[len("negative:"):].strip()

    # if any of the concepts is empty, raise an error
    if not positive_concept1 or not positive_concept2 or not negative_concept:
        raise ValueError("Failed to extract all concepts from the LLM output.")

    return positive_concept1, positive_concept2, negative_concept

def process_llm_output_paraphrasing(output_text):
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
        >>> process_llm_output_paraphrasing('Rephrased Caption: "This image shows a cat."\nOther content here.')
        'This image shows a cat.'
        
        >>> process_llm_output_paraphrasing('"A dog is sitting on the grass."\nNote: The grass is green.')
        'A dog is sitting on the grass.'

        >>> process_llm_output_paraphrasing('This is a simple caption.')
        'This is a simple caption.'

        >>> process_llm_output_paraphrasing('Rephrased Caption: This is another caption.\nFurther explanation follows.')
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

def process_concepts_task(df, model_name, args, max_attempts=3):
    """
    Processes the task by generating two positive concepts and one negative concept for each set of original captions in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data, including a column for original captions.
        model_name (str): The full model name to be used for the LLM.
        args (argparse.Namespace): Parsed command-line arguments.
        max_attempts (int): The maximum number of attempts to generate a valid output for each video.
    """
    llm, sampling_params = initialize_llm(model_name)

    # Initialize new columns to store the generated concepts
    df["positive_concept1"] = None
    df["positive_concept2"] = None
    df["negative_concept"] = None

    start_time = time.time()

    for i in tqdm(range(args.index_start, args.index_end), desc="Generating concepts"):
        original_captions = eval(df.loc[i, "all_captions"])  # Assuming captions are stored as a string representation of a list

        attempts = 0
        success = False

        while attempts < max_attempts and not success:
            attempts += 1
            try:
                # Randomly select and shuffle three examples
                selected_examples = random.sample(concepts_examples_list, 3)
                random.shuffle(selected_examples)
                examples = "\n\n".join(selected_examples)

                # Generate a prompt for the LLM based on the original captions and examples
                prompt = generate_prompt("concepts", original_captions, examples)

                # Generate the concepts using the LLM
                output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]

                generated_text = output.outputs[0].text
                positive_concept1, positive_concept2, negative_concept = process_llm_output_concepts(generated_text)
                
                # Store the results in the DataFrame
                df.loc[i, "positive_concept1"] = positive_concept1
                df.loc[i, "positive_concept2"] = positive_concept2
                df.loc[i, "negative_concept"] = negative_concept

                # Append or prepend the negative concept to the original captions "There is no {negative_concept} in the video."
                # 50% chance to append, 50% chance to prepend to every caption in the list
                captions = eval(df.loc[i, "captions"])
                for caption_index in range(len(captions)):
                    if random.random() < 0.5:
                        captions[caption_index] = f"{captions[caption_index]}. There is no {negative_concept} in the video."
                    else:
                        captions[caption_index] = f"There is no {negative_concept} in the video. {captions[caption_index]}."

                df.loc[i, "captions"] = str(captions)  # Update the captions in the DataFrame

                success = True  # Mark as successful if no exception occurs

            except Exception as e:
                print(f"Attempt {attempts} failed for index {i}. Error: {e}")

        if not success:
            print(f"Failed to generate output for index {i} after {max_attempts} attempts.")

    # Reorder the columns in the following order: "image_id", "captions", "positive_concept1", "positive_concept2", "negative_concept", "all_captions", "filepath"
    df = df[["image_id", "captions", "positive_concept1", "positive_concept2", "negative_concept", "all_captions", "filepath"]]

    end_time = time.time()
    print(f"Generation time (for {args.index_end - args.index_start} entries): {end_time - start_time} seconds")

def process_paraphrasing_task(df, model_name, args, paraphrasing_examples, use_affirmation_negation_guideline, dataset_type='mcq', max_attempts=3):
    """
    Processes the task by generating paraphrased captions for each caption in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data. For 'mcq' type, with columns 'caption_0', 'caption_1', 'caption_2', 'caption_3'.
                           For 'retrieval' type, captions are stored in the 'captions' column as a list of strings.
        model_name (str): The full model name to be used for the LLM.
        args (argparse.Namespace): Parsed command-line arguments.
        paraphrasing_examples (list): A list of example prompts and responses to guide the LLM's output.
        dataset_type (str): The type of dataset, either 'mcq' or 'retrieval'.
        max_attempts (int): The maximum number of attempts to generate a valid output for each caption.
        use_affirmation_negation_guideline (bool): Flag to control whether to include the affirmation/negation order guideline in the paraphrasing prompt.
    """
    llm, sampling_params = initialize_llm(model_name)

    print(f"Using affirmation/negation guideline: {use_affirmation_negation_guideline}")

    start_time = time.time()

    for i in tqdm(range(args.index_start, args.index_end), desc="Paraphrasing captions"):
        if dataset_type == 'mcq':
            captions = [df.loc[i, f"caption_{j}"] for j in range(4)]
        elif dataset_type == 'retrieval':
            captions = eval(df.loc[i, "captions"])  # Assuming captions are stored as a string representation of a list

        new_captions = []

        for caption in captions:
            attempts = 0
            success = False

            while attempts < max_attempts and not success:
                attempts += 1
                try:
                    # Randomly select and shuffle five examples
                    selected_examples = random.sample(paraphrasing_examples, 5)
                    random.shuffle(selected_examples)
                    examples = "\n\n".join(selected_examples)

                    # Generate a prompt for the LLM based on the original caption, examples, and the guideline flag
                    prompt = generate_prompt_paraphrasing(caption, examples, use_affirmation_negation_guideline)

                    # Generate the paraphrased caption using the LLM
                    output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]

                    generated_text = output.outputs[0].text
                    rephrased_caption = process_llm_output_paraphrasing(generated_text)
                    
                    # Store the rephrased caption
                    new_captions.append(rephrased_caption)

                    success = True  # Mark as successful if no exception occurs

                except Exception as e:
                    print(f"Attempt {attempts} failed for index {i}. Caption: {caption} Error: {e}")

            if not success:
                print(f"Failed to generate output for index {i}, Caption: {caption} after {max_attempts} attempts.")
                print(f"Using the original caption without paraphrasing.")
                new_captions.append(caption)  # Append the original caption if all attempts fail

        if dataset_type == 'mcq':
            for j in range(4):
                df.loc[i, f"caption_{j}"] = new_captions[j]
        elif dataset_type == 'retrieval':
            df.loc[i, "captions"] = str(new_captions)

    end_time = time.time()
    print(f"Paraphrasing time (for {args.index_end - args.index_start} entries): {end_time - start_time} seconds")

def main(args):
    # Map model aliases to full model names
    model_mapping = {
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # "mixtral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama3.1": "meta-llama/Meta-Llama-3.1-70B-Instruct"
    }
    
    model_name = model_mapping[args.model]  # Use alias to get full model name

    df = pd.read_csv(args.input_file)

    # Set index_end to the length of the DataFrame if it's -1 or exceeds the length of the DataFrame
    if args.index_end == -1 or args.index_end > len(df):
        args.index_end = len(df)

    # Subset the DataFrame to only the rows specified by index_start and index_end
    df_subset = df.iloc[args.index_start:args.index_end].copy()

    if args.task == "concepts":
        # Process the task of generating concepts only
        process_concepts_task(df_subset, model_name, args)
    elif args.task == "retrieval":
        process_paraphrasing_task(df_subset, model_name, args, paraphrasing_retrieval_examples_list, 
                                  dataset_type='retrieval', use_affirmation_negation_guideline=args.use_affirmation_negation_guideline)
    elif args.task == "mcq":
        process_paraphrasing_task(df_subset, model_name, args, paraphrasing_mcq_examples_list, 
                                  dataset_type='mcq', use_affirmation_negation_guideline=args.use_affirmation_negation_guideline)
    else: # raise an error if the task is not recognized
        raise ValueError(f"Invalid task type: {args.task}")

    # Generate output file name based on the processed range
    if args.index_start == 0 and args.index_end == len(df):
        output_file_name = f"{args.output_base}.csv"
    else:
        output_file_name = f"{args.output_base}_{args.index_start}_{args.index_end}.csv"

    # Save the modified DataFrame with the new negative concepts and captions
    df_subset.to_csv(output_file_name, index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)
