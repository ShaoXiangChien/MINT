"""
This script processes a test dataset to create multiple-choice questions (MCQs) for each image or video, depending on the specified task.

### Functionality:
1. **Input CSV File**:
   - The input file should include the following columns:
     - For images:
       - `positive_objects`: A list of positive objects (e.g., `["object1", "object2"]`).
       - `negative_objects`: A list of negative objects (e.g., `["object3", "object4"]`).
       - `filepath`: Path to the image file.
     - For videos:
       - `positive_concept1`: A positive concept (e.g., `"concept1"`).
       - `positive_concept2`: Another positive concept (e.g., `"concept2"`).
       - `negative_concept`: A negative concept (e.g., `"concept3"`).
       - `filepath`: Path to the video file.

2. **Output CSV File**:
   - The output file contains the following columns:
     - `correct_answer`: Index of the correct answer (always 0 for implementation simplicity).
     - `caption_0` to `caption_3`: Four answer choices, including the correct one.
     - `correct_answer_template`: The template used to generate the correct answer.
     - `image_path`: Path to the corresponding image or video.

3. **Templates**:
   - For constructing MCQs, the following templates are used:
     - **Positive**: "This [image/video] features {A} and {B}."
     - **Negative**: "This [image/video] does not feature {N}."
     - **Hybrid**: "This [image/video] features {B}, but not {N}."

4. **Answer Shuffling**:
   - The correct answer is always placed as the first choice (`caption_0`) for simplicity.
   - For an embedding-based model like CLIP, the order of answer choices does not impact the evaluation.

### Example Usage:
#### For Images:
```bash
python create_mcq.py \
    --task image \
    --input_file data/images/test_image_data.csv \
    --output_file data/images/test_image_mcq.csv
```
This creates an MCQ dataset for images and saves the output to `test_image_mcq.csv`.

#### For Videos:
```bash
python create_mcq.py \
    --task video \
    --input_file data/videos/test_video_data.csv \
    --output_file data/videos/test_video_mcq.csv
```
This creates an MCQ dataset for videos and saves the output to `test_video_mcq.csv`.

### Notes:
- The number of questions generated for each image is limited by the number of available negative objects (up to 3).
- For videos, one question is generated per row in the input file.
- Ensure the input CSV file is correctly formatted with the required columns before running the script.
"""

import pandas as pd
import random
import argparse

# Define the command-line arguments
parser = argparse.ArgumentParser() # input_file, output_file, task
parser.add_argument('--task', type=str, default='image', help='The task to perform: image, synthetic_image, image_single_positive, image_no_negation, video')
parser.add_argument('--input_file', type=str, help='The path to the input test CSV file')
parser.add_argument('--output_file', type=str, help='The path where the output CSV file will be saved')
args = parser.parse_args()

def create_video_mcq_dataframe(test_csv, output_csv):
    """
    This function takes in a CSV file test_csv containing test data, processes it to create a 
    new DataFrame mcq_df with multiple-choice questions, and saves it to output_csv.

    The correct_answer column holds the index of the correct caption based on the chosen template.

    Parameters:
    - test_csv (str): The path to the input test CSV file.
    - output_csv (str): The path where the output CSV file will be saved.
    """
    # Load the test CSV file into a DataFrame
    test_df = pd.read_csv(test_csv)

    # Initialize a list to store the rows for the new mcq_df DataFrame
    mcq_data = []

    # Define the possible templates
    templates = ["positive", "negative", "hybrid"]

    # Loop over each row in the test DataFrame
    for _, row in test_df.iterrows():
        A = row['positive_concept1']
        B = row['positive_concept2']
        N = row['negative_concept']
        filepath = row['filepath']

        # Randomly choose the template
        right_template = random.choice(templates)

        # Construct the right answer based on the chosen template
        if right_template == "positive":
            right_answer = f"This video features {A} and {B}"
        elif right_template == "negative":
            right_answer = f"This video does not feature {N}"
        elif right_template == "hybrid":
            right_answer = f"This video features {B}, but not {N}"

        # Construct the three wrong answers
        wrong_answer_1 = f"This video features {N}, but not {B}"
        wrong_answer_2 = f"This video features {N}"
        wrong_answer_3 = f"This video does not feature {A}"

        # Randomly shuffle the correct and wrong answers
        answers = [right_answer, wrong_answer_1, wrong_answer_2, wrong_answer_3]
        correct_answer_index = answers.index(right_answer)

        # Append the constructed row to mcq_data
        mcq_data.append({
            'image_path': filepath,
            'correct_answer': correct_answer_index,
            'caption_0': answers[0],
            'caption_1': answers[1],
            'caption_2': answers[2],
            'caption_3': answers[3],
            'correct_answer_template': right_template
        })

    # Create the mcq_df DataFrame from the mcq_data list
    mcq_df = pd.DataFrame(mcq_data)

    # Reorder columns to match the expected output: correct_answer,caption_0,caption_1,caption_2,caption_3,correct_answer_template,image_path
    mcq_df = mcq_df[['correct_answer', 'caption_0', 'caption_1', 'caption_2', 'caption_3', 'correct_answer_template', 'image_path']]

    # Save the mcq_df DataFrame to a CSV file
    mcq_df.to_csv(output_csv, index=False)

    print(f'Video MCQ dataset saved to {output_csv}')

def create_image_mcq_dataframe(test_csv, output_csv):
    """
    This function creates a DataFrame of multiple-choice questions for image-based VQA.
    It handles up to 3 unique questions per image based on available positive and negative objects.
    
    The correct answer is always placed first (index 0), with other answers shuffled.
    
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

    # Loop over each row in the test DataFrame
    for _, row in test_df.iterrows():
        positive_objects = eval(row['positive_objects'])
        negative_objects = eval(row['negative_objects'])
        filepath = row['filepath']

        # Ensure there are at least 2 positive objects for constructing A and B
        if len(positive_objects) < 2:
            continue  # Skip the row if not enough positive objects for A and B

        # Determine the number of questions to generate, based on negative objects (up to 3)
        num_questions = min(len(negative_objects), 3)

        # Randomly sample 2 positive objects for A and B
        A, B = random.sample(positive_objects, 2)

        # Sample N (negative concepts) without replacement for each question
        sampled_negatives = random.sample(negative_objects, num_questions)

        # Generate one unique question for each sampled negative object (N)
        for i, N in enumerate(sampled_negatives):
            # Randomly choose the template
            right_template = random.choice(templates)

            # Construct the correct answer based on the chosen template
            if right_template == "positive":
                right_answer = f"This image features {A} and {B}"
            elif right_template == "negative":
                right_answer = f"This image does not feature {N}"
            elif right_template == "hybrid":
                right_answer = f"This image features {B}, but not {N}"

            # Construct the three wrong answers
            wrong_answer_1 = f"This image features {N}, but not {B}"
            wrong_answer_2 = f"This image features {N}"
            wrong_answer_3 = f"This image does not feature {A}"

            # The correct answer is always placed at index 0
            answers = [right_answer, wrong_answer_1, wrong_answer_2, wrong_answer_3]

            # Append the constructed row to mcq_data
            mcq_data.append({
                'image_path': filepath,
                'correct_answer': 0,  # Correct answer is always at index 0
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

    print(f'Number of questions generated was: {len(mcq_df)}')

# main 
if __name__ == '__main__':
    if args.task == 'image':
        create_image_mcq_dataframe(args.input_file, args.output_file)
    elif args.task == 'video':
        create_video_mcq_dataframe(args.input_file, args.output_file)
    else:
        raise ValueError(f'Invalid task: {args.task}. Please choose either "image" or "video".')