#!/bin/bash

# DataTrove preprocessing script
# Equivalent to the original Preprocess.bash but using DataTrove

# Set input parameters
DATASET_NAME="emozilla/dolma-v1_7-books"     # HuggingFace dataset name
DATASET_CONFIG="default"                     # Dataset configuration
NUM_SAMPLES_TOTAL=100000000                      # Total samples to load from dataset
TOKENIZER_NAME="meta-llama/Llama-3.2-3B"    # Tokenizer for sliding window
WINDOW_SIZE=16384                            # Window size for sliding window
SAMPLE_SIZE=100000000                             # Final sample size
PREFIX="sample_"                             # Data ID prefix
OUTPUT_FOLDER="/fsx/jason/LongAttn/datatrove_output_dolma_16k"  # Output folder
NUM_WORKERS=64                                # Number of workers (DataTrove handles parallelization differently)

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Run the DataTrove preprocessing pipeline
python datatrove_preprocessing.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_config "$DATASET_CONFIG" \
    --num_samples_total "$NUM_SAMPLES_TOTAL" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --window_size "$WINDOW_SIZE" \
    --sample_size "$SAMPLE_SIZE" \
    --prefix "$PREFIX" \
    --output_folder "$OUTPUT_FOLDER" \
    --num_workers "$NUM_WORKERS"

if [ $? -eq 0 ]; then
    echo "DataTrove preprocessing completed successfully!"
    echo "Output saved to: $OUTPUT_FOLDER/preprocessed_data.jsonl"
    echo "Logs saved to: $OUTPUT_FOLDER/logs/"
else
    echo "DataTrove preprocessing failed, please check the error messages."
fi 