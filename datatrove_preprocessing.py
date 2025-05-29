#!/usr/bin/env python3
"""
DataTrove pipeline for preprocessing text data with sliding window sampling.
Replicates the functionality of the original preprocessing pipeline.
"""

import os
import json
import random
import argparse
from typing import Iterator, Optional, List
from datasets import load_dataset
from transformers import AutoTokenizer

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor


class HuggingFaceReader(PipelineStep):
    """Custom reader that loads data from HuggingFace datasets."""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_config: str = "default",
        split: str = "train", 
        text_field: str = "text",
        num_samples: Optional[int] = None,
        streaming: bool = True
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.num_samples = num_samples
        self.streaming = streaming
        
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Load data from HuggingFace dataset."""
        print(f"Loading dataset '{self.dataset_name}'...")
        
        # Load the dataset
        dataset = load_dataset(
            self.dataset_name, 
            name=self.dataset_config,
            split=self.split,
            streaming=self.streaming
        )
        
        count = 0
        for example in dataset:
            if self.num_samples and count >= self.num_samples:
                break
                
            # Only process on the appropriate rank for distributed processing
            if count % world_size == rank:
                if example and self.text_field in example and example[self.text_field]:
                    doc = Document(
                        text=example[self.text_field],
                        id=f"doc_{count:08d}"
                    )
                    yield doc
                    
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} entries...")


class SlidingWindowFilter(PipelineStep):
    """Custom filter that applies sliding window sampling to documents."""
    
    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Llama-3.2-3B",
        window_size: int = 32768
    ):
        super().__init__()
        self.window_size = window_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def sliding_window_sample(self, data: List[int], window_size: int) -> List[List[int]]:
        """
        Slice from both ends of the list towards the middle, cutting a window from 
        the front and back each time, until the length of the middle segment falls 
        within a specified range.
        """
        data_length = len(data)
        segments = []

        if data_length < window_size:
            return segments

        left = 0
        right = data_length

        while (right - left) > 3 * window_size:
            # cutting a window from left
            segments.append(data[left:left + window_size])
            left += window_size

            # cutting a window from right  
            segments.append(data[right - window_size:right])
            right -= window_size

        remaining_length = right - left

        if 1 * window_size < remaining_length <= 2 * window_size:
            # if middle length is between window_size and 2 * window_size 
            segments.append(data[left:left + window_size])
            segments.append(data[right - window_size:right])
        elif 2 * window_size < remaining_length <= 3 * window_size:
            # if middle length is between 2 * window_size and 3 * window_size
            segments.append(data[left:left + window_size])
            middle_start = left + (remaining_length - window_size) // 2
            segments.append(data[middle_start:middle_start + window_size])
            segments.append(data[right - window_size:right])

        return segments
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Apply sliding window sampling to each document."""
        for document in data:
            with self.track_time():
                # Tokenize the text
                token_ids = self.tokenizer.encode(document.text)
                
                # Remove the first token (typically BOS token)
                if len(token_ids) > 0:
                    token_ids = token_ids[1:]
                
                # Apply sliding window sampling
                segments = self.sliding_window_sample(token_ids, self.window_size)
                
                # Decode segments back to text and yield as separate documents
                if segments:
                    decoded_segments = self.tokenizer.batch_decode(segments)
                    for i, segment_text in enumerate(decoded_segments):
                        segment_doc = Document(
                            text=segment_text,
                            id=f"{document.id}_segment_{i}",
                            metadata=document.metadata.copy() if document.metadata else {}
                        )
                        self.stat_update("segments_created", 1)
                        yield segment_doc
                else:
                    # If no segments created, document was too short
                    self.stat_update("documents_too_short", 1)


class SamplerWithIDFilter(PipelineStep):
    """Custom filter that samples data and assigns unique IDs."""
    
    def __init__(
        self,
        sample_size: Optional[int] = None,
        prefix: str = "sample_",
        seed: int = 12
    ):
        super().__init__()
        self.sample_size = sample_size
        self.prefix = prefix
        self.seed = seed
        random.seed(seed)
        
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Sample documents and assign unique IDs."""
        # First, collect all documents
        documents = list(data)
        
        print(f"Total documents before sampling: {len(documents)}")
        
        # Sample if needed
        if self.sample_size and self.sample_size < len(documents):
            sampled_docs = random.sample(documents, self.sample_size)
            print(f"Sampled {len(sampled_docs)} documents")
        else:
            sampled_docs = documents
            print(f"Using all {len(sampled_docs)} documents (sample_size >= total)")
        
        # Assign unique IDs and yield
        for i, doc in enumerate(sampled_docs, start=1):
            doc.id = f"{self.prefix}{str(i).zfill(7)}"
            doc.metadata = doc.metadata or {}
            doc.metadata["data_id"] = doc.id
            self.stat_update("final_documents", 1)
            yield doc


def create_datatrove_pipeline(
    dataset_name: str = "emozilla/dolma-v1_7-books",
    dataset_config: str = "default",
    num_samples_total: int = 1000,
    tokenizer_name: str = "meta-llama/Llama-3.2-3B",
    window_size: int = 32768,
    sample_size: int = 1000,
    prefix: str = "sample_",
    output_folder: str = "/fsx/jason/LongAttn/datatrove_output"
):
    """Create the DataTrove pipeline."""
    
    pipeline = [
        HuggingFaceReader(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            text_field="text",
            num_samples=num_samples_total,
            streaming=False
        ),
        SlidingWindowFilter(
            tokenizer_name=tokenizer_name,
            window_size=window_size
        ),
        SamplerWithIDFilter(
            sample_size=sample_size,
            prefix=prefix
        ),
        JsonlWriter(
            output_folder=output_folder,
            output_filename="preprocessed_data_${rank}.jsonl",
            compression=None
        )
    ]
    
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="DataTrove preprocessing pipeline")
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default="emozilla/dolma-v1_7-books",
                        help='HuggingFace dataset name')
    parser.add_argument('--dataset_config', type=str, default="default",
                        help='Dataset configuration')
    parser.add_argument('--num_samples_total', type=int, default=1000,
                        help='Total number of samples to load from dataset')
    
    # Processing arguments
    parser.add_argument('--tokenizer_name', type=str, default="meta-llama/Llama-3.2-3B",
                        help='Tokenizer to use for sliding window')
    parser.add_argument('--window_size', type=int, default=32768,
                        help='Window size for sliding window sampling')
    
    # Sampling arguments
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of final samples to extract')
    parser.add_argument('--prefix', type=str, default='sample_',
                        help='Prefix for unique data IDs')
    
    # Output arguments
    parser.add_argument('--output_folder', type=str, default="/fsx/jason/LongAttn/datatrove_output",
                        help='Output folder for processed data')
    
    # Execution arguments
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for parallel processing')
    
    args = parser.parse_args()

    print("Prefetching dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train", streaming=False)
    print("Dataset prefetched!")
    
    # Create the pipeline
    pipeline = create_datatrove_pipeline(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_samples_total=args.num_samples_total,
        tokenizer_name=args.tokenizer_name,
        window_size=args.window_size,
        sample_size=args.sample_size,
        prefix=args.prefix,
        output_folder=args.output_folder
    )
    
    # Execute the pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=args.num_workers,
        logging_dir=os.path.join(args.output_folder, "logs")
    )
    
    print("Starting DataTrove preprocessing pipeline...")
    executor.run()
    print("DataTrove preprocessing completed!")


if __name__ == "__main__":
    main() 