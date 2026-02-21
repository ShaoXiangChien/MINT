import os
import sys
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/DeepSeek-VL2")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/DeepSeek-VL2")

from PIL import Image
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from general_utils import ModelAndTokenizer, prepare_inputs
import json
from datasets import load_from_disk
from tqdm import tqdm


def load_vlm(device):
    """
    Load the DeepSeek vision-language model.
    This function is identical to the original experiment for consistency.
    """
    # Load vision-language model
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    mt = ModelAndTokenizer(
        "deepseek-ai/deepseek-vl2-7b-tiny",
        model=vl_gpt,
        tokenizer=tokenizer,
        processor=vl_chat_processor,
        low_cpu_mem_usage=False,
        device=device
    )
    mt.model = mt.model.to(mt.device)
    return mt


def generate_baseline_response(mt, prompt, image, max_gen_len=20, temperature=None, verbose=False):
    """
    Generate a baseline response from the vision-language model WITHOUT any patching.
    
    This is the key difference from the original experiment:
    - No layer-wise patching or inspection
    - Direct generation from the model using the given image and prompt
    - Mimics standard VLM inference behavior
    
    Args:
        mt: ModelAndTokenizer object containing the model, tokenizer, and processor
        prompt: Text prompt to send to the model
        image: PIL Image to process
        max_gen_len: Maximum number of tokens to generate
        temperature: Sampling temperature (None for greedy decoding)
        verbose: Whether to print debug information
        
    Returns:
        Generated text response from the model
    """
    # Get the device from the model
    device = next(mt.model.parameters()).device
    
    # Prepare inputs using the same method as the original experiment
    # This ensures consistency in input preprocessing
    inputs = prepare_inputs(prompt, image.resize((384, 384)), mt.processor, device)
    
    # Generate response without any hooks or patching
    # This is pure model inference - what the model naturally produces
    with torch.no_grad():
        output_ids = mt.model.generate(
            inputs_embeds=mt.model.prepare_inputs_embeds(**inputs),
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_gen_len,
            pad_token_id=mt.tokenizer.pad_token_id,
            eos_token_id=mt.tokenizer.eos_token_id,
            bos_token_id=mt.tokenizer.bos_token_id,
            temperature=temperature if temperature is not None else 1.0,
            do_sample=(temperature is not None and temperature > 0)
        )
    
    # Decode the generated tokens to text
    output_text = mt.processor.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    if verbose:
        print(f"Generated baseline output: {output_text}")
    
    return output_text


def load_cate_mapping(mapping_path):
    """
    Load category mapping from JSON file.
    This function is identical to the original experiment.
    """
    with open(mapping_path, "r") as f:
        category_map = json.load(f)
    return category_map


if __name__ == "__main__": # Set device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Output directory for baseline results
    result_dir = "/home/sc305/VLM/Final Experiments/global-img-ld/lg_results/ds-baseline-results.json"

    # Load dataset and category mapping (same as original)
    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(device)
    
    # Load target image (this seems to be unused in baseline, but kept for consistency)
    target_image = Image.open("./images/apple.png")

    exp_results = []
    pbar = tqdm(sample_trainset)
    count = 0

    print("Starting baseline experiment...")
    print("This experiment generates responses without any layer patching or inspection.")
    print("Each sample will be processed once, representing the model's natural behavior.")
    
    for sample in pbar:
        # Get category for this sample
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"What is in the image? {category} or apple"

        try:
            # BASELINE APPROACH: Single response generation without patching
            # Unlike the original experiment which tests different layer combinations,
            # the baseline simply asks the model what it sees in the image
            result = generate_baseline_response(mt, prompt, sample["image"])
            
            # Determine if the model correctly identified the category
            # This uses the same evaluation logic as the original experiment
            prediction = 1 if category in result.lower() else 0
            
            # Store results in a format that's comparable to the original experiment
            # Note: For baseline, we don't have layer-wise results, so we store a single prediction
            exp_results.append({
                "sample_id": count,
                "category": category,
                "prompt": prompt,
                "model_response": result,
                "prediction": prediction,
                "baseline": True  # Flag to indicate this is baseline data
            })
            
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            continue
        
        # Save results after each sample (same approach as original)
        with open(result_dir, "w") as f:
            json.dump(exp_results, f, indent=2)

        count += 1
        pbar.set_description(f"Processed: {count}, Correct predictions: {sum(r['prediction'] for r in exp_results)}")

    # Final results summary
    total_samples = len(exp_results)
    correct_predictions = sum(r['prediction'] for r in exp_results)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print(f"\n=== BASELINE EXPERIMENT RESULTS ===")
    print(f"Total samples processed: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Baseline accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Save final results
    with open(result_dir, "w") as f:
        json.dump(exp_results, f, indent=2)
    
    print(f"Results saved to: {result_dir}")
