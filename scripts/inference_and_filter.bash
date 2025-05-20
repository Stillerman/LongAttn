# #!/bin/bash

# # Set input parameters for the first script
file_path="/fsx/jason/LongAttn/sample_output.jsonl"  # Input file path
output_file="/fsx/jason/LongAttn/output.jsonl"  # Output file path for inference
batch_size=6  # Batch size

# Check batch_size
if ! [[ "$batch_size" =~ ^[0-9]+$ ]]; then
    echo "Error: Batch size must be a number."
    exit 1
fi

# Define error log file
ERROR_LOG="error.log"

# Clear the existing error log file
> "$ERROR_LOG"

# Run the first inference script
echo "Running inference..."
if ! accelerate launch src/1_inference_dp.py "$file_path" "$output_file" "$batch_size" 2>>"$ERROR_LOG"; then
    echo "Inference script failed. Check $ERROR_LOG for details."
    exit 1
fi

# Define parameters for the second script (DateSorted)
FILE_PATH="$file_path"
INFERENCE_PATH="$output_file"
OUTPUT_PATH="/fsx/jason/LongAttn/final_output.jsonl"

# Run the second script (DateSorted)
echo "Running DateSorted processing..."
if ! python src/2_filtering_by_Lds.py \
    --inference_path $INFERENCE_PATH \
    --output_path $OUTPUT_PATH \
    --file_path $FILE_PATH 2>>"$ERROR_LOG"; then
    echo "DateSorted processing script failed. Check $ERROR_LOG for details."
    exit 1
fi

echo "Processing complete. Results saved to $OUTPUT_PATH."
