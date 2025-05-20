#!/bin/bash

# Set input parameters
INPUT_FOLDER="/fsx/jason/LongAttn/input"          # Input folder path
OUTPUT_FOLDER="/fsx/jason/LongAttn/output"        # Output folder path
MERGED_FILE="/fsx/jason/LongAttn/merged_file.jsonl" # Path to the merged file
SAMPLE_OUTPUT="/fsx/jason/LongAttn/sample_output.jsonl" # Path to the sampled output file
SAMPLE_SIZE=1000                       # Sample size
PREFIX="sample_"                       # Data ID prefix
MAX_WORKERS=100                         # Maximum number of worker threads
BATCH_SIZE=500                         # Batch size
WINDOW_SIZE=32768                      # Window size

# Run the Python script
python src/0_pre-process.py \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --max_workers "$MAX_WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --window_size "$WINDOW_SIZE" \
    --merged_file "$MERGED_FILE" \
    --sample_output "$SAMPLE_OUTPUT" \
    --sample_size "$SAMPLE_SIZE" \
    --prefix "$PREFIX"

if [ $? -eq 0 ]; then
    echo "Data processing completed successfully!"
else
    echo "Data processing failed, please check the error messages."
fi
