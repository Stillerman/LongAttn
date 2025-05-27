import os
from datasets import load_dataset
import jsonlines

def create_gutenberg_input_file(output_dir="input", output_filename_base="gutenberg_input", num_samples_total=30000, num_samples_per_file=3000):
    """
    Downloads the first num_samples_total from the 'manu/project_gutenberg' dataset
    and saves them into multiple JSONL files in the specified output directory.
    Each file will contain num_samples_per_file entries (except possibly the last file).
    Each line in the output files will be a JSON object: {"content": "text_from_dataset"}.
    """
    print(f"Loading dataset 'manu/project_gutenberg'...")
    # Load the dataset, streaming if possible for large datasets, though for 500 it's fine.
    # The 'default' config is often used for this dataset.
    dataset = load_dataset("emozilla/dolma-v1_7-books", name="default", split="train", streaming=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {num_samples_total} entries and writing to multiple files...")
    
    count = 0
    file_index = 0
    current_file_count = 0
    writer = None
    
    for example in dataset:
        if count >= num_samples_total:
            break
            
        # Open a new file when needed
        if current_file_count == 0:
            if writer:
                writer.close()
            output_filename = f"{output_filename_base}_{file_index:03d}.jsonl"
            output_path = os.path.join(output_dir, output_filename)
            print(f"Writing to file: {output_path}")
            writer = jsonlines.open(output_path, mode='w')
            file_index += 1
            
        if example and 'text' in example and example['text']: # Ensure text is not empty
            writer.write({"content": example['text']})
            count += 1
            current_file_count += 1
            
            # Reset counter when we reach samples per file limit
            if current_file_count >= num_samples_per_file:
                current_file_count = 0
                
            if count % 50 == 0:
                print(f"Processed {count}/{num_samples_total} entries...")
                
    # Close the last writer if open
    if writer:
        writer.close()
                
    if count < num_samples_total:
        print(f"Warning: Only able to retrieve {count} entries, which is less than the requested {num_samples_total}.")
    print(f"Successfully created {file_index} files with a total of {count} entries.")

if __name__ == "__main__":
    create_gutenberg_input_file(output_dir="/fsx/jason/LongAttn/input", num_samples_total=1000, num_samples_per_file=1000)
    print("Dataset preparation script finished.")
