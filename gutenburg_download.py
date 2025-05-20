import os
from datasets import load_dataset
import jsonlines

def create_gutenberg_input_file(output_dir="input", output_filename="gutenberg_input.jsonl", num_samples=3000):
    """
    Downloads the first num_samples from the 'manu/project_gutenberg' dataset
    and saves them into a JSONL file in the specified output directory.
    Each line in the output file will be a JSON object: {"content": "text_from_dataset"}.
    """
    print(f"Loading dataset 'manu/project_gutenberg'...")
    # Load the dataset, streaming if possible for large datasets, though for 500 it's fine.
    # The 'default' config is often used for this dataset.
    dataset = load_dataset("manu/project_gutenberg", name="default", split="en", streaming=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Processing the first {num_samples} entries and writing to {output_path}...")
    
    count = 0
    with jsonlines.open(output_path, mode='w') as writer:
        for example in dataset:
            if count >= num_samples:
                break
            if example and 'text' in example and example['text']: # Ensure text is not empty
                writer.write({"content": example['text']})
                count += 1
            if count % 50 == 0 and count > 0:
                print(f"Processed {count}/{num_samples} entries...")
                
    if count < num_samples:
        print(f"Warning: Only able to retrieve {count} entries, which is less than the requested {num_samples}.")
    print(f"Successfully created {output_path} with {count} entries.")

if __name__ == "__main__":
    
    create_gutenberg_input_file(output_dir="/fsx/jason/LongAttn/input", num_samples=3000)
    print("Dataset preparation script finished.")
