"""
This script processes images and captions to verify that specified negative objects are not detected in the images. 
It uses the OWL-ViT model to detect objects and updates the list of negative objects in a CSV file by removing 
those that are detected in the corresponding images.

### Functionality:
1. **Input CSV File**:
   - Must contain the following columns:
     - `filepath`: Path to the image file.
     - `negative_objects`: A list of negative objects to check against the image.
   - The `negative_objects` column should be a string representation of a list (e.g., `["object1", "object2"]`).

2. **Processing**:
   - For each image, the script checks whether the objects in the `negative_objects` column are detected.
   - Detected objects are removed from the list in the `negative_objects` column.

3. **Output CSV File**:
   - Includes all columns from the input file, with the updated `negative_objects` column reflecting the remaining undetected objects.

### Example Usage:
#### Basic Example:
```bash
python filter_negative_objects.py \
    --input_file image_captions_with_objects.csv \
    --output_file updated_image_captions.csv
```
This processes all rows in `image_captions_with_objects.csv` and saves the updated file as `updated_image_captions.csv`.

#### Processing a Chunk of Rows:
```bash
python filter_negative_objects.py \
    --input_file image_captions_with_objects.csv \
    --output_file updated_image_captions_chunk.csv \
    --start_index 0 \
    --end_index 1000
```
This processes rows 0 to 999 in `image_captions_with_objects.csv` and saves the results in `updated_image_captions_chunk.csv`.

#### Batch Size:
To adjust the number of images processed in a single batch:
```bash
python filter_negative_objects.py \
    --input_file image_captions_with_objects.csv \
    --output_file updated_image_captions.csv \
    --batch_size 32
```

### Notes:
- Use `--start_index` and `--end_index` to process large datasets in chunks.
- If `--end_index` is `-1`, the script processes all rows from `--start_index` to the end of the file.
- The output reflects the updated list of verified negative objects for each image.
"""

from transformers import OwlViTProcessor, OwlViTForObjectDetection
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

def parse_args():
    """
    Parse command-line arguments for the script.

    Returns:
        args: The parsed arguments containing input and output file paths.
    """
    parser = argparse.ArgumentParser(description='Filter detected negative objects from images and update the CSV file.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file with filepaths, captions, extracted objects, and negative objects.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the updated CSV file.')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index of the chunk to process (inclusive). Default is 0.')
    parser.add_argument('--end_index', type=int, default=-1, help='Ending index of the chunk to process (exclusive). Default is -1 (process all rows from start_index).')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images to process in a batch. Default is 128.')
    
    args = parser.parse_args()
    return args

def load_image(filepath):
    """
    Load an image from the specified filepath.

    Args:
        filepath (str): The path to the image file.

    Returns:
        Image: The loaded image in RGB format, or None if the image cannot be loaded.
    """
    try:
        image = Image.open(filepath).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None
    
def process_negative_objects(owl_detector, processor, images, texts, device):
    """
    Detect and filter negative objects from a batch of images.

    Args:
        owl_detector: The OWL-ViT object detector model.
        processor: The OWL-ViT processor for preprocessing and postprocessing.
        images (list): A list of images to process.
        texts (list): A list of lists of negative objects to verify against the images.
        device: The device (CPU or GPU) to perform the computation on.

    Returns:
        results (list): A list of lists containing the negative objects detected in each image.
    """
    # Construct the text prompts with the "A photo of " prefix
    text_prompts = [["A photo of " + obj for obj in objects] for objects in texts]
    
    # Process the inputs with the constructed text prompts
    inputs = processor(text=text_prompts, images=images, return_tensors="pt").to(device)
    outputs = owl_detector(**inputs)
    target_sizes = torch.Tensor([img.size[::-1] for img in images]).to(device)
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    detected_objects_batch = []
    score_threshold = 0.1

    for i in range(len(images)):
        detected_objects_set = set()
        scores, labels = results[i]["scores"], results[i]["labels"]

        for score, label in zip(scores, labels):
            if score >= score_threshold:
                # Add the original object label instead of the prefixed prompt
                detected_objects_set.add(texts[i][label])

        detected_objects_batch.append(list(detected_objects_set))

    return detected_objects_batch

def main(args):
    """
    Main function to load the CSV file, process images, and filter detected negative objects.

    Args:
        args: The parsed command-line arguments.
    """
    # Determine the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CSV file
    df = pd.read_csv(args.input_file)
    
    # Select the chunk to process
    if args.end_index == -1:
        chunk = df.iloc[args.start_index:]
    else:
        chunk = df.iloc[args.start_index:args.end_index]
    print(f"Length of chunk: {len(chunk)}")

    # Initialize the open-vocabulary detector
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_detector = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    # Initialize counters for negative objects
    total_negative_objects = 0
    total_remaining_objects = 0
    images_with_all_negative_removed = 0

    # Initialize batch containers
    batch_images = []
    batch_indices = []
    batch_negative_objects = []

    # Process the chunk in batches
    for index, row in tqdm(chunk.iterrows(), desc="Processing images", total=len(chunk)):
        filepath = row['filepath']
        negative_objects = eval(row['negative_objects'])  # Convert string representation of list back to list
        if not negative_objects or len(negative_objects) != 5:  # Skip if there are no negative objects or if the number is not 5
            continue

        image = load_image(filepath)
        if image is None:
            print(f"Error loading image {filepath}. Skipping...")
            continue

        # Accumulate batch data
        batch_images.append(image)
        batch_indices.append(index)
        batch_negative_objects.append(negative_objects)

        # Process the batch if the batch size is reached
        if len(batch_images) == args.batch_size:
            try:
                results = process_negative_objects(owl_detector, processor, batch_images, batch_negative_objects, device)
                total_negative_objects += sum(len(objs) for objs in batch_negative_objects)  # Count only after successful processing
                # Update the DataFrame with the remaining negative objects
                for i, detected_objects in enumerate(results):
                    remaining_negative_objects = [obj for obj in batch_negative_objects[i] if obj not in detected_objects]
                    chunk.at[batch_indices[i], 'negative_objects'] = str(remaining_negative_objects)
                    total_remaining_objects += len(remaining_negative_objects)

                    if len(remaining_negative_objects) == 0:
                        images_with_all_negative_removed += 1

            except Exception as e:
                print(f"Error processing batch: {e}")
                batch_images, batch_indices, batch_negative_objects = [], [], []  # Clear the batch
                continue

            # Clear the batch
            batch_images, batch_indices, batch_negative_objects = [], [], []

    # Process any remaining images in the final batch
    if batch_images:
        print("Processing final batch...")
        try:
            results = process_negative_objects(owl_detector, processor, batch_images, batch_negative_objects, device)
            total_negative_objects += sum(len(objs) for objs in batch_negative_objects)  # Count only after successful processing
            
            for i, detected_objects in enumerate(results):
                remaining_negative_objects = [obj for obj in batch_negative_objects[i] if obj not in detected_objects]
                chunk.at[batch_indices[i], 'negative_objects'] = str(remaining_negative_objects)
                total_remaining_objects += len(remaining_negative_objects)

                if len(remaining_negative_objects) == 0:
                    images_with_all_negative_removed += 1
        except Exception as e:
            print(f"Error processing final batch: {e}")

    # Calculate the number of removed objects and their percentage
    removed_objects = total_negative_objects - total_remaining_objects
    removed_percentage = (removed_objects / total_negative_objects) * 100 if total_negative_objects > 0 else 0

    # Calculate percentage of images where all negative objects were removed
    total_images = len(chunk)
    all_removed_percentage = (images_with_all_negative_removed / total_images) * 100 if total_images > 0 else 0

    # Print the results
    print(f"Chunk processed: {args.start_index} to {args.start_index + len(chunk)}")
    print(f"Total images processed: {total_images}")
    print(f"Total negative objects: {total_negative_objects}")
    print(f"Total removed objects: {removed_objects}")
    print(f"Percentage of removed objects: {removed_percentage:.2f}%")
    print(f"Images with all negative objects removed: {images_with_all_negative_removed}")
    print(f"Percentage of images with all negative objects removed: {all_removed_percentage:.2f}%")

    # Save the updated DataFrame to the output CSV file
    chunk.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
