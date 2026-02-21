import os
import torch
import warnings
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants for image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height), Image.LANCZOS)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size), Image.LANCZOS)
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def main():
    # Configuration
    # NOTE: Verify the model ID. 'InternVL3' might be a placeholder or very new.
    # If it fails, try 'OpenGVLab/InternVL2-40B' or similar if that's what you have access to.
    model_id = "OpenGVLab/InternVL3-38B-Instruct" 
    device_map = "auto" # Allows using multiple GPUs
    dtype = torch.bfloat16
    
    # Sample Image (You can change this path)
    image_path = "./sample_image.jpg"
    
    # Check if image exists, if not create a dummy one
    if not os.path.exists(image_path):
        print(f"Creating dummy image at {image_path}")
        dummy_img = Image.new('RGB', (800, 600), color='red')
        dummy_img.save(image_path)

    print(f"Loading model: {model_id}")
    print(f"Precision: {dtype}")
    
    # Check for Flash Attention
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        attn_impl = "eager" # or "sdpa" if torch version allows, but eager is safest fallback
        print("Flash Attention not found, falling back to default (eager). Install flash-attn for better performance.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_impl
    ).eval()

    print("Model loaded successfully!")
    
    # Load and preprocess image
    pixel_values = load_image(image_path, max_num=12).to(dtype).cuda()
    
    # Prepare generation config
    generation_config = dict(
        num_beams=1,
        max_new_tokens=1024,
        do_sample=False,
    )

    # Basic Inference
    question = "Please describe this image in detail."
    print(f"Question: {question}")
    
    # Calling model.chat (Standard for InternVL models)
    # Note: The chat method signature might vary slightly between versions.
    # Usually: tokenizer, pixel_values, question, generation_config
    # InternVL2 often returns only the response string directly in newer implementations
    try:
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        # Handle case where it might return (response, history) tuple
        if isinstance(response, tuple):
             response = response[0]
        
        print("-" * 50)
        print("Response:")
        print(response)
        print("-" * 50)
    except AttributeError:
        print("model.chat method not found. Trying model.generate...")
        # Fallback if chat is not available (generic transformers usage)
        # This part depends heavily on how the model was trained/wrapped
        inputs = tokenizer(question, return_tensors="pt").to("cuda")
        # NOTE: This fallback likely won't work well for VLM without pixel_values handling
        # specific to the model architecture.
        outputs = model.generate(**inputs, max_new_tokens=200)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

