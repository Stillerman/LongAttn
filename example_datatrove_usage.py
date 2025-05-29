#!/usr/bin/env python3
"""
Example usage of the DataTrove preprocessing pipeline.
"""

import os
from datatrove_preprocessing import create_datatrove_pipeline
from datatrove.executor import LocalPipelineExecutor


def run_preprocessing_example():
    """Run a small example of the preprocessing pipeline."""
    
    # Configure the pipeline
    pipeline = create_datatrove_pipeline(
        dataset_name="emozilla/dolma-v1_7-books",
        dataset_config="default",
        num_samples_total=10,  # Small number for testing
        tokenizer_name="meta-llama/Llama-3.2-3B",
        window_size=32768,
        sample_size=5,  # Even smaller for final output
        prefix="example_",
        output_folder="./example_output"
    )
    
    # Create output directory
    os.makedirs("./example_output", exist_ok=True)
    
    # Execute the pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir="./example_output/logs"
    )
    
    print("Running DataTrove preprocessing example...")
    executor.run()
    print("Example completed! Check ./example_output/preprocessed_data.jsonl")


if __name__ == "__main__":
    run_preprocessing_example() 