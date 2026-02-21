"""
This script processes images and captions to verify that the specified negative objects are not detected in the images. 
It uses the OWL-ViT model to detect objects and updates the list of negative objects in the input CSV file 
by removing any objects that are detected in the corresponding images.

### Input CSV Requirements:
The input CSV file should contain the following columns:
- `filepath`: Path to the image file.
- `negative_objects`: A list of negative objects associated with the image.

### Output CSV:
The output CSV will include the updated `negative_objects` column, with any detected negative objects removed.

### Example Usage:

#### Filter Negative Objects:
```bash
python filter_negative_objects.py \
    --input_file path/to/input.csv \
    --output_file path/to/output_filtered.csv \
    --start_index 0 \
    --end_index 1000 \
    --batch_size 64
```

#### Chunk Processing (if the dataset is large):
Process different chunks of the dataset sequentially:
```bash
python filter_negative_objects.py \
    --input_file path/to/input.csv \
    --output_file path/to/output_filtered_0_1000.csv \
    --start_index 0 \
    --end_index 1000 \
    --batch_size 64

python filter_negative_objects.py \
    --input_file path/to/input.csv \
    --output_file path/to/output_filtered_1000_2000.csv \
    --start_index 1000 \
    --end_index 2000 \
    --batch_size 64
```

### Notes:
1. **Batch Processing**: 
   - Images are processed in batches to optimize GPU utilization. The batch size can be adjusted using the `--batch_size` argument.
   
2. **Negative Objects**:
   - Ensure that the `negative_objects` column contains valid lists. Rows with empty or malformed lists will be skipped.

3. **GPU Requirement**:
   - The OWL-ViT model requires GPU resources for efficient processing. Ensure a compatible GPU is available.

4. **Output Naming Convention**:
   - Use descriptive names for output files to reflect the processing stage (e.g., `output_filtered_0_1000.csv` for the first chunk).

5. **Edge Cases**:
   - Images that cannot be loaded or rows with no valid negative objects are skipped during processing.

### Output Summary:
At the end of processing, the script prints a summary that includes:
- Total images processed.
- Total negative objects.
- Percentage of removed negative objects.
- Percentage of images where all negative objects were removed.
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
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images to process in a batch. Default is 64.')
    
    args = parser.parse_args()
    return args
# Commands to reproduce results:
# python filter_negative_objects.py --end_index 128 --input_file /data/healthy-ml/gobi1/data/cc3m/negation_dataset/train_images_extracted_pos_neg_0_294525.csv --output_file /data/healthy-ml/gobi1/data/cc3m/negation_dataset/train_images_extracted_pos_neg_filtered_0_294525_64.csv

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
