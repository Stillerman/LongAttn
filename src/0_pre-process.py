import os
import argparse
from DataProcess import SlideWindow, FileMerger, sample_data
from transformers import AutoTokenizer

def main():

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

    parser = argparse.ArgumentParser(description="Process and sample data using SlideWindow, FileMerger, and sample_data.")
    
    # Arguments for SlideWindow
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing raw data files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder for processed data.')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of workers for parallel processing.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing.')
    parser.add_argument('--window_size', type=int, default=32768, help='Window size for sliding window sampling.')

    # Arguments for FileMerger
    parser.add_argument('--merged_file', type=str, required=True, help='Path to the output merged file.')

    # Arguments for sample_data
    parser.add_argument('--sample_output', type=str, required=True, help='Path to the output sampled file.')
    parser.add_argument('--sample_size', type=int, required=True, help='Number of samples to extract.')
    parser.add_argument('--prefix', type=str, default='sample_', help='Prefix for unique data IDs.')

    args = parser.parse_args()

    # Step 1: Process data using SlideWindow
    slide_window = SlideWindow(llm_tokenizer=tokenizer, max_workers=args.max_workers, batch_size=args.batch_size, window_size=args.window_size)
    slide_window.data_part(args.input_folder, args.output_folder)

    # Step 2: Merge processed files using FileMerger
    file_merger = FileMerger(folder_path=args.output_folder, output_file=args.merged_file)
    file_merger.merge_files()

    # Step 3: Sample data and assign unique IDs
    sample_data(file_path=args.merged_file, output_path=args.sample_output, prefix=args.prefix, sample_size=args.sample_size)

if __name__ == "__main__":
    main()