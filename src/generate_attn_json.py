import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
from accelerate import Accelerator


def generate_attention_json(text_path: str, output_json_path: str, model_name: str = "Qwen/Qwen2.5-1.5B"):
    """
    Generates attention weights for the first layer of a Transformer model
    for a given text file and saves it as a JSON.

    Args:
        text_path (str): Path to the input text file.
        output_json_path (str): Path to save the output JSON file.
        model_name (str): Name of the Hugging Face model to use (e.g., "distilbert-base-uncased").
        max_seq_len (int): Maximum sequence length to process. Text will be truncated if longer.
                           Common models like BERT/DistilBERT often have a max of 512 tokens.
    """
    try:
        # Initialize accelerator with mixed precision
        accelerator = Accelerator(mixed_precision="fp16")
        print(f"Using device: {accelerator.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using AutoModelForMaskedLM as it's common for models that output attentions and often
        # matches the architecture for base models. For a pure encoder, AutoModel is also fine.
        model = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            load_in_8bit=True,
            )
        
        # Move model to the device managed by accelerator
        model = accelerator.prepare(model)
        model.eval() # Set model to evaluation mode

        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found at: {text_path}")

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize the text, adding special tokens and truncating to max_seq_len
        # `add_special_tokens=True` ensures [CLS] and [SEP] tokens are included,
        # which are part of the attention computation .
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        
        # Move inputs to the device managed by accelerator
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        # Get the actual tokens, including special tokens (e.g., '[CLS]', '[SEP]', '##word')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        with torch.no_grad():
            outputs = model(**inputs)

        # `outputs.attentions` is a tuple of (num_layers) tensors.
        # Each tensor has shape (batch_size, num_heads, seq_len, seq_len).
        # We only care about the first layer's attention (index 0) and the first (and only) batch item (index 0).
        first_layer_attention = outputs.attentions[0][0].to("cpu") # Move to CPU for numpy conversion

        # Convert the attention tensor to a list of lists for JSON serialization
        attention_weights_list = first_layer_attention.numpy().tolist()

        # Only have rank 0 process save the output
        if accelerator.is_main_process:
            data = {
                "tokens": tokens,
                "attention_weights": attention_weights_list
            }

            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            print(f"Attention map generated and saved to {output_json_path}")
            print(f"Number of tokens: {len(tokens)}")
            print(f"Number of heads: {len(attention_weights_list)}")
            if attention_weights_list:
                print(f"Shape of first layer attention weights: {len(attention_weights_list)} heads, {len(attention_weights_list[0])} query tokens, {len(attention_weights_list[0][0])} key tokens")
            else:
                print("No attention weights generated (perhaps input text was too short or model issue).")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'transformers' and 'torch' libraries installed.")
        print("You can install them using: pip install transformers torch")


if __name__ == "__main__":
    # Read firstline of jsonl file
    with open("/fsx/jason/LongAttn/final_output.jsonl", "r", encoding="utf-8") as f:
        first_line = f.readline()
        first_line_json = json.loads(first_line)
        first_line_text = first_line_json["content"]
    
    
    dummy_text_path = "sample_long_text.txt"
    with open(dummy_text_path, "w", encoding="utf-8") as f:
        f.write(first_line_text.strip())

    # Generate the attention JSON using the dummy text
    # This will create 'attention_map.json' in the same directory.
    generate_attention_json(dummy_text_path, "attention_map.json")
    print("Done")
