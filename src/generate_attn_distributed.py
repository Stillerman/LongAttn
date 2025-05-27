import os
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
import json
import jsonlines
from tqdm import tqdm
import time
from accelerate import Accelerator
from LongAttn import CustomLlamaForCausalLM

def generate_attention_json(text, model, tokenizer, device):
    """
    Generates attention weights for a given text using the custom LongAttn model.
    
    Args:
        text (str): The input text.
        model: The model to use.
        tokenizer: The tokenizer to use.
        device: The device to run inference on.
    
    Returns:
        dict: Dictionary containing tokens and attention weights.
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    
    # Move inputs to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get the actual tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Run the model with attention output enabled
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )
    
    # Extract first layer attention weights
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    first_layer_attention = outputs.attentions[0][0].cpu().numpy().tolist()
    
    return {
        "tokens": tokens,
        "attention_weights": first_layer_attention
    }

def process_jsonl_file(data_list, tokenizer, model, device, output_dir, batch_size=1):
    """
    Process a list of data items from a JSONL file and generate attention maps.
    
    Args:
        data_list (list): List of data items to process.
        tokenizer: The tokenizer to use.
        model: The model to use.
        device: The device to use.
        output_dir (str): Directory to save attention maps.
        batch_size (int): Batch size for processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    bos_token = tokenizer.bos_token
    batch = []
    data_ids = []
    
    for data in tqdm(data_list, desc=f"Processing on {device}"):
        batch.append(bos_token + ' ' + data['content'])
        data_ids.append(data['data_id'])
        
        if len(batch) >= batch_size:
            for i, (text, data_id) in enumerate(zip(batch, data_ids)):
                try:
                    attention_data = generate_attention_json(text, model, tokenizer, device)
                    output_file = os.path.join(output_dir, f"{data_id}_attention.json")
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(attention_data, f)
                    
                    print(f"Saved attention map for {data_id} to {output_file}")
                    print(f"Number of tokens: {len(attention_data['tokens'])}")
                    print(f"Attention shape: {len(attention_data['attention_weights'])} heads, "
                          f"{len(attention_data['attention_weights'][0])} query tokens, "
                          f"{len(attention_data['attention_weights'][0][0])} key tokens")
                except Exception as e:
                    print(f"Error processing {data_id}: {e}")
            
            batch = []
            data_ids = []
    
    # Process any remaining items
    if batch:
        for i, (text, data_id) in enumerate(zip(batch, data_ids)):
            try:
                attention_data = generate_attention_json(text, model, tokenizer, device)
                output_file = os.path.join(output_dir, f"{data_id}_attention.json")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(attention_data, f)
                
                print(f"Saved attention map for {data_id} to {output_file}")
            except Exception as e:
                print(f"Error processing {data_id}: {e}")

def process_text_file(text_path, output_json_path, tokenizer, model, device):
    """
    Process a single text file and generate an attention map.
    This function is compatible with the original generate_attn_json.py script.
    
    Args:
        text_path (str): Path to the input text file.
        output_json_path (str): Path to save the output JSON file.
        tokenizer: The tokenizer to use.
        model: The model to use.
        device: The device to use.
    """
    try:
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found at: {text_path}")

        # Read the text file
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"Processing text file: {text_path}")
        
        # Generate the attention data
        attention_data = generate_attention_json(text, model, tokenizer, device)
        
        # Save the attention data to the output JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(attention_data, f, indent=2)

        print(f"Attention map generated and saved to {output_json_path}")
        print(f"Number of tokens: {len(attention_data['tokens'])}")
        print(f"Number of heads: {len(attention_data['attention_weights'])}")
        if attention_data['attention_weights']:
            print(f"Shape of first layer attention weights: {len(attention_data['attention_weights'])} heads, "
                  f"{len(attention_data['attention_weights'][0])} query tokens, "
                  f"{len(attention_data['attention_weights'][0][0])} key tokens")
        else:
            print("No attention weights generated (perhaps input text was too short or model issue).")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'transformers' and 'torch' libraries installed.")

def main():
    parser = argparse.ArgumentParser(description='Generate attention maps, either for a text file or in distributed manner.')
    
    # Add mode argument to determine how to process
    parser.add_argument('--mode', type=str, choices=['text', 'jsonl'], default='text',
                        help='Mode to run in: text (single file) or jsonl (distributed)')
    
    # Arguments for text file mode (compatible with original script)
    parser.add_argument('--text_path', type=str, help='Path to the input text file (for text mode)')
    parser.add_argument('--output_json_path', type=str, default="attention_map.json",
                        help='Path to save the output JSON file (for text mode)')
    
    # Arguments for JSONL mode (distributed processing)
    parser.add_argument('--input_jsonl', type=str, help='Path to the input JSONL file (for jsonl mode)')
    parser.add_argument('--output_dir', type=str, help='Directory to save attention maps (for jsonl mode)')
    
    # Shared arguments
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B", 
                        help='HuggingFace model name/path')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size per GPU (for jsonl mode, use 1 for very long contexts)')
    parser.add_argument('--max_length', type=int, default=32768,
                        help='Maximum sequence length to process')
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Set up model configuration
    config_kwargs = {
        "rope_theta": 2500000.0,  # For longer context
    }
    
    config = AutoConfig.from_pretrained(args.model_name, **config_kwargs)
    config.num_hidden_layers = 1  # Only need first layer for attention maps
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length
    
    # Load the model using the custom implementation
    model = CustomLlamaForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map={"": accelerator.process_index}
    )
    model.eval()
    
    start_time = time.time()
    
    # Process based on mode
    if args.mode == 'text':
        # Compatibility mode with original script
        # Only use the main process for this mode
        if accelerator.is_main_process:
            if not args.text_path:
                # Read from first line of jsonl file for compatibility with original script
                jsonl_path = "/fsx/jason/LongAttn/final_output.jsonl"
                dummy_text_path = "sample_long_text.txt"
                
                if os.path.exists(jsonl_path):
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        first_line = f.readline()
                        first_line_json = json.loads(first_line)
                        first_line_text = first_line_json["content"]
                    
                    with open(dummy_text_path, "w", encoding="utf-8") as f:
                        f.write(first_line_text.strip())
                    
                    args.text_path = dummy_text_path
                else:
                    raise ValueError("Please provide a text file path with --text_path")
            
            process_text_file(
                args.text_path, 
                args.output_json_path, 
                tokenizer, 
                model, 
                device
            )
    else:  # args.mode == 'jsonl'
        # Distributed processing mode
        if not args.input_jsonl or not args.output_dir:
            raise ValueError("For jsonl mode, please provide --input_jsonl and --output_dir")
        
        # Load data
        with jsonlines.open(args.input_jsonl) as reader:
            data_list = [obj for obj in reader]
        
        # Split data between processes
        with accelerator.split_between_processes(data_list) as data_parts:
            print(f"Processing {len(data_parts)} samples on device {device}, "
                  f"process index: {accelerator.process_index}")
            
            # Process the data
            process_jsonl_file(
                data_parts, 
                tokenizer, 
                model, 
                device, 
                args.output_dir,
                batch_size=args.batch_size
            )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time on {device}: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 