"""
create_mcq.py: Generates multiple-choice questions (MCQs) for image-based visual question answering (VQA).

### Functionality:
- Processes a CSV file containing extracted positive and negative objects for images.
- Generates MCQs with one correct answer and three incorrect answers based on the provided objects.
- Supports three templates:
  1. Positive: "This image features A and B."
  2. Negative: "This image does not feature N."
  3. Hybrid: "This image features A, but not N."
- Handles cases with fewer than two positive objects using a modified positive template.

### Example Usage:
```bash
python create_mcq.py --task image --input_file path/to/input.csv --output_file path/to/output.csv
```

### Input:
- A CSV file with columns:
  - `filepath`: Path to the image.
  - `extracted_objects`: List of positive objects.
  - `negative_objects`: List of negative objects.

### Output:
- A CSV file with columns:
  - `correct_answer`: Index of the correct caption (0-3).
  - `caption_0`, `caption_1`, `caption_2`, `caption_3`: Answer choices.
  - `correct_answer_template`: Template type (positive, negative, hybrid, etc.).
  - `image_path`: Path to the image.

### Notes:
- Skips rows with no negative objects.
- Randomly shuffles answers to ensure varied placements of the correct answer.
"""

import pandas as pd
import random
import argparse
from tqdm import tqdm  # Import tqdm for progress tracking

# Define the command-line arguments
parser = argparse.ArgumentParser()  # input_file, output_file, task
parser.add_argument('--task', type=str, default='image', help='The task to perform: image')
parser.add_argument('--input_file', type=str, help='The path to the input test CSV file')
parser.add_argument('--output_file', type=str, help='The path where the output CSV file will be saved')
args = parser.parse_args()

def create_image_mcq_dataframe(test_csv, output_csv):
    """
    This function creates a DataFrame of multiple-choice questions for image-based VQA.
    It generates one question per image based on available positive and negative objects,
    and shuffles the answers so the correct answer is not always at index 0.
    
    Parameters:
    - test_csv (str): The path to the input test CSV file containing image data.
    - output_csv (str): The path where the output CSV file will be saved.
    """
    # Load the test CSV file into a DataFrame
    test_df = pd.read_csv(test_csv)

    # Initialize a list to store the rows for the new mcq_df DataFrame
    mcq_data = []

    # Define the possible templates
    templates = ["positive", "negative", "hybrid"]

    # Loop over each row in the test DataFrame with tqdm for progress tracking
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing rows"):
        positive_objects = eval(row['extracted_objects'])
        negative_objects = eval(row['negative_objects'])
        filepath = row['filepath']

        # If less than 2 positive objects, restrict template choices to "negative" and a modified "positive"
        if len(positive_objects) < 2:
            A = positive_objects[0] if len(positive_objects) == 1 else None  # If 1 positive object, use it
            allowed_templates = ["negative", "hybrid", "modified_positive"]
        else:
            # If enough positive objects, allow all templates and sample two positive objects
            A, B = random.sample(positive_objects, 2)
            allowed_templates = templates

        # Sample one negative object (N) if available
        if len(negative_objects) == 0:
            continue  # Skip rows with no negative objects

        N = random.choice(negative_objects)

        # Randomly choose the template
        right_template = random.choice(allowed_templates)

        # Construct the correct answer based on the chosen template
        if right_template == "positive":
            right_answer = f"This image features {A} and {B}."
        elif right_template == "negative":
            right_answer = f"This image does not feature {N}."
        elif right_template == "hybrid":
            right_answer = f"This image features {A}, but not {N}."
        elif right_template == "modified_positive":
            right_answer = f"This image features {A}."  # For cases with fewer than 2 positive objects

        # Construct three wrong answers
        wrong_answer_1 = f"This image features {N}, but not {A}."
        wrong_answer_2 = f"This image features {N}."
        wrong_answer_3 = f"This image does not feature {A}."

        # Collect all answers
        answers = [right_answer, wrong_answer_1, wrong_answer_2, wrong_answer_3]

        # Shuffle the answers and keep track of where the correct answer ends up
        random.shuffle(answers)
        correct_answer_idx = answers.index(right_answer)

        # Append the constructed row to mcq_data
        mcq_data.append({
            'image_path': filepath,
            'correct_answer': correct_answer_idx,  # Track the correct answer index after shuffling
            'caption_0': answers[0],
            'caption_1': answers[1],
            'caption_2': answers[2],
            'caption_3': answers[3],
            'correct_answer_template': right_template
        })

    # Create the mcq_df DataFrame from the mcq_data list
    mcq_df = pd.DataFrame(mcq_data)

    # Reorder columns to match the expected output
    mcq_df = mcq_df[['correct_answer', 'caption_0', 'caption_1', 'caption_2', 'caption_3', 'correct_answer_template', 'image_path']]

    # Save the mcq_df DataFrame to a CSV file
    mcq_df.to_csv(output_csv, index=False)

    print(f'Image MCQ dataset saved to {output_csv}')
    print(f'Number of questions generated: {len(mcq_df)}')

# main 
if __name__ == '__main__':
    if args.task == 'image':
        create_image_mcq_dataframe(args.input_file, args.output_file)
    else:
        raise ValueError(f'Invalid task: {args.task}. Please choose either "image".')
