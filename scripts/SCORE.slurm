#!/bin/bash
#SBATCH --job-name=score-shards
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --begin=now+0minutes
#SBATCH --output=slurm_out/grader-%j.out
#SBATCH --time=00:45:00
#SBATCH --array=0-63%4

set -e -x

source /fsx/jason/LongAttn/.venv/bin/activate

# print start time
echo "Starting at $(date)"

# 1 shard per job
base=$((SLURM_ARRAY_TASK_ID))

shard_name=$(printf "%05d" $base)

# # Set input parameters for the first script
file_path="/fsx/jason/LongAttn/datatrove_output_dolma_16k/preprocessed_data_${shard_name}.jsonl"  # Input file path
output_file="/fsx/jason/LongAttn/datatrove_output_dolma_16k/inference_output_${shard_name}.jsonl"  # Output file path for inference
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
OUTPUT_PATH="/fsx/jason/LongAttn/datatrove_output_dolma_16k/final_output_${shard_name}.jsonl"

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


# print start time
echo "DONE at $(date)"