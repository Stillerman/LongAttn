# run_datatrove_pipeline.py

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import HeadFilter
from custom_steps import SlidingWindowSegmenter, FixedSamplerAndIdAssigner, CustomModelInference, TokenDistanceScoreFilter # Import new step

# Define parameters
HF_DATASET_NAME = "manu/project_gutenberg"
HF_DATASET_SPLIT = "en"
NUM_SAMPLES_DOWNLOAD = 3000
OUTPUT_BASE_DIR = "output_gutenberg_pipeline"

# Intermediate logging/data paths - outputs are no longer explicitly written by JsonlWriters at these stages
# LOGGING_DIR_02_SEGMENTATION = f"{OUTPUT_BASE_DIR}/logs/02_segmentation"
# DATA_DIR_02_SEGMENTED = f"{OUTPUT_BASE_DIR}/02_segmented_data" 
# LOGGING_DIR_03_SAMPLED_ID = f"{OUTPUT_BASE_DIR}/logs/03_sampled_id"
# DATA_DIR_03_SAMPLED_ID = f"{OUTPUT_BASE_DIR}/03_sampled_id_data"
# LOGGING_DIR_04_INFERENCE = f"{OUTPUT_BASE_DIR}/logs/04_inference"
# DATA_DIR_04_INFERENCE_OUTPUT = f"{OUTPUT_BASE_DIR}/04_inference_output_data"

TOKENIZER_PATH_SLIDING_WINDOW = "meta-llama/Llama-3.2-3B"
WINDOW_SIZE_SLIDING_WINDOW = 32768

SAMPLING_SIZE_AFTER_SEGMENTATION = 1000
ID_ASSIGNER_PREFIX = "sample_"

INFERENCE_MODEL_PATH = "meta-llama/Llama-3.1-70B" 
INFERENCE_BATCH_SIZE = 6 

# Logging and data directories for Step 5: Final Filtering
LOGGING_DIR_05_FILTERED = f"{OUTPUT_BASE_DIR}/logs/05_final_filtered"
FINAL_OUTPUT_DIR = f"{OUTPUT_BASE_DIR}/final_filtered_data" # Final output directory

# Note on PYTHONPATH for custom model:
# Ensure 'src' directory (containing LongAttn.py) is in PYTHONPATH.
# Example: export PYTHONPATH=$PYTHONPATH:$(pwd)/src
# Or, if your script is in project root and src is a subdirectory:
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    pipeline = [
        HuggingFaceDatasetReader(
            dataset=HF_DATASET_NAME,
            dataset_options={"split": HF_DATASET_SPLIT},
            text_key="text" 
        ),
        HeadFilter(limit=NUM_SAMPLES_DOWNLOAD),
        SlidingWindowSegmenter(
            tokenizer_path=TOKENIZER_PATH_SLIDING_WINDOW,
            window_size=WINDOW_SIZE_SLIDING_WINDOW
        ),
        FixedSamplerAndIdAssigner(
            sample_size=SAMPLING_SIZE_AFTER_SEGMENTATION,
            id_prefix=ID_ASSIGNER_PREFIX
        ),
        CustomModelInference(
            model_path=INFERENCE_MODEL_PATH,
            batch_size=INFERENCE_BATCH_SIZE
        ),
        # JsonlWriter for inference output removed
        TokenDistanceScoreFilter(
            # top_percentage_to_keep defaults to 0.5, matching original script
        ),
        JsonlWriter(
            output_folder=FINAL_OUTPUT_DIR, # Final output after filtering
            output_filename="${rank}.jsonl.gz" 
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=LOGGING_DIR_05_FILTERED, # Updated logging directory for the final stage
        tasks=1, # Crucial for global sampling/filtering steps
        workers=1 # Crucial for global sampling/filtering steps
    )
    executor.run()

if __name__ == "__main__":
    main()
