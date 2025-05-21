# custom_steps.py
import logging # Added import for logger
import random # Add this import
import math # Add this import
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document, DocumentsPipeline
from transformers import AutoTokenizer, AutoConfig # Added AutoConfig
import torch # Added torch

logger = logging.getLogger(__name__) # Added logger instance

# Attempt to import CustomLlamaForCausalLM
# This assumes that LongAttn.py is in a directory structure like src/LongAttn.py
# and the script running datatrove is at a level where 'src' is importable.
try:
    from src.LongAttn import CustomLlamaForCausalLM
except ModuleNotFoundError:
    logger.error("Could not import CustomLlamaForCausalLM from src.LongAttn. "
                 "Ensure LongAttn.py is in the 'src' directory and 'src' is a package or accessible via PYTHONPATH.")
    CustomLlamaForCausalLM = None
except ImportError as e:
    logger.error(f"ImportError when importing CustomLlamaForCausalLM: {e}. Check dependencies for LongAttn.py.")
    CustomLlamaForCausalLM = None


class SlidingWindowSegmenter(PipelineStep):
    type = "🪟 SLIDING_WINDOW"
    _requires_dependencies = ["transformers"]

    def __init__(self, tokenizer_path: str, window_size: int, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.tokenizer = None

    def get_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self.tokenizer

    def _sliding_window_sample_logic(self, token_ids):
        data_length = len(token_ids)
        segments = []
        window_size = self.window_size

        if data_length < window_size:
            return segments

        left = 0
        right = data_length

        while (right - left) > 3 * window_size:
            segments.append(token_ids[left:left + window_size])
            left += window_size
            segments.append(token_ids[right - window_size:right])
            right -= window_size
        
        remaining_length = right - left

        if 1 * window_size < remaining_length <= 2 * window_size:
            segments.append(token_ids[left:left + window_size])
            segments.append(token_ids[right - window_size:right])
        elif 2 * window_size < remaining_length <= 3 * window_size:
            segments.append(token_ids[left:left + window_size])
            middle_start = left + (remaining_length - window_size) // 2
            segments.append(token_ids[middle_start:middle_start + window_size])
            segments.append(token_ids[right - window_size:right])
        
        return segments

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        tokenizer = self.get_tokenizer()
        for doc in data:
            self.stat_update("documents_in")
            try:
                token_ids = tokenizer.encode(doc.text)
                
                processed_token_ids = []
                if len(token_ids) > 1:
                    processed_token_ids = token_ids[1:]
                elif len(token_ids) == 1: # If only one token (e.g. BOS), then slicing [1:] makes it empty
                    processed_token_ids = []
                # If token_ids is empty list, processed_token_ids remains empty

                if not processed_token_ids:
                    self.stat_update("empty_after_tokenization_or_slice")
                    # Original script would not produce segments, effectively dropping.
                    # We continue to next doc.
                    continue

                text_segments_tokens = self._sliding_window_sample_logic(processed_token_ids)

                if not text_segments_tokens:
                    self.stat_update("no_segments_produced")
                    # No segments produced, continue to next doc.
                    continue

                # tokenizer.batch_decode can fail on empty list of lists
                if not any(text_segments_tokens): # check if all sublists are empty or the main list is empty
                     decoded_segments = []
                else:
                     decoded_segments = tokenizer.batch_decode(text_segments_tokens, skip_special_tokens=True)


                for segment_text in decoded_segments:
                    if not segment_text.strip(): # Skip empty segments
                        self.stat_update("empty_decoded_segments_skipped")
                        continue
                    new_doc = Document(text=segment_text, metadata=doc.metadata.copy() if doc.metadata else {})
                    if doc.id: # Ensure doc.id exists
                        new_doc.metadata["original_doc_id"] = doc.id
                    self.stat_update("segments_produced")
                    yield new_doc
            except Exception as e:
                self.stat_update("exceptions")
                logger.error(f"Error processing document ID {doc.id if doc.id else 'Unknown'} in SlidingWindowSegmenter: {e}", exc_info=True)
                # Yielding original doc on error might be problematic if subsequent steps expect segmented format.
                # For now, let's not yield it to maintain data integrity for later stages.
                # If a doc fails segmentation, it's better it's excluded from segmented data.
                continue # go to next document

class FixedSamplerAndIdAssigner(PipelineStep):
    type = "🎯 FIXED_SAMPLER_ID"
    
    def __init__(self, sample_size: int, id_prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.sample_size = sample_size
        self.id_prefix = id_prefix
        # Counter for IDs is managed per call to run, effectively per task/shard.
        # For global unique IDs from a counter, this step should run as a single task.

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if world_size > 1:
            logger.warning(f"{self.type} step is currently designed to best ensure a global fixed sample size "
                           f"and sequential IDs when run as a single task (tasks=1 in executor). "
                           f"Running with world_size={world_size}, rank={rank}. "
                           f"Sampling will be per-task, and IDs will be unique per task if prefix includes rank, "
                           f"or potentially overlap if not and multiple tasks yield IDs.")
        
        collected_docs = []
        for doc in data:
            collected_docs.append(doc)
        self.stat_update("documents_input_for_sampling", value=len(collected_docs))

        if self.sample_size >= len(collected_docs):
            sampled_docs = collected_docs
            self.stat_update("sampling_took_all")
        else:
            sampled_docs = random.sample(collected_docs, self.sample_size)
            self.stat_update("sampling_random_sample_executed")
        
        self.stat_update("documents_sampled_count", value=len(sampled_docs))

        # ID generation: starts from 1 for each task's processed batch of sampled docs.
        # To match original script's single counter for all 1000 samples, this step
        # should ideally be run with tasks=1 in the executor.
        current_id_in_task = 0
        for doc_idx, doc in enumerate(sampled_docs):
            current_id_in_task += 1 
            # The original script's data_id format is "prefix" + "0000001"
            # Example: sample_0000001
            unique_id = f"{self.id_prefix}{current_id_in_task:07d}"
            
            if doc.metadata is None: # Ensure metadata exists
                doc.metadata = {}
            doc.metadata["data_id"] = unique_id
            # also store it with the step name as key, for datatrove stats/convention
            # self.stat_update(self.name, value=unique_id) # Store last generated ID for example. Removed as per discussion.
            
            self.stat_update("documents_id_assigned")
            yield doc

class CustomModelInference(PipelineStep):
    type = "🧠 CUSTOM_INFERENCE"
    _requires_dependencies = ["torch", "transformers"] # Assuming src.LongAttn's deps are covered

    def __init__(self, model_path: str, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.device = None
        self.rank = 0 # Will be set in run
        self.world_size = 1 # Will be set in run

    def _initialize_model_and_tokenizer(self):
        if self.tokenizer is None: # Initialize once
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.model is None: # Initialize once
            if CustomLlamaForCausalLM is None:
                raise RuntimeError("CustomLlamaForCausalLM could not be loaded. Cannot proceed with inference.")

            config_kwargs = {
                "rope_theta": 2500000.0, # From original 1_inference_dp.py script
            }
            # trust_remote_code=True might be needed for AutoConfig as well if model has custom code
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, **config_kwargs)
            config.num_hidden_layers = 1 # Crucial modification

            self.model = CustomLlamaForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=True 
            )
            self.model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"CustomModelInference (rank {self.rank}, world {self.world_size}) initialized model on device: {self.device}")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        # Store rank and world_size for _initialize_model_and_tokenizer if needed for device assignment
        self.rank = rank
        self.world_size = world_size
        self._initialize_model_and_tokenizer() # Ensures model is loaded once per task

        batch_texts = []
        batch_metadata = []

        for doc in data:
            self.stat_update("documents_in")
            if not doc.text or not doc.text.strip():
                self.stat_update("skipped_empty_text")
                yield doc # Pass through if no text
                continue
            
            batch_texts.append(doc.text)
            batch_metadata.append(doc.metadata)

            if len(batch_texts) >= self.batch_size:
                for processed_doc in self._process_batch(batch_texts, batch_metadata):
                    yield processed_doc
                batch_texts = []
                batch_metadata = []
        
        if batch_texts: # Process any remaining documents
            for processed_doc in self._process_batch(batch_texts, batch_metadata):
                yield processed_doc

    def _process_batch(self, text_batch, metadata_batch):
        try:
            # BOS token is typically added by LlamaTokenizers if add_special_tokens=True
            tokens = self.tokenizer.batch_encode_plus(
                text_batch,
                add_special_tokens=True, 
                padding='max_length',
                truncation=True,
                max_length=32768, # From original script's process_batch
                return_tensors='pt'
            )
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Assuming outputs.proportions and outputs.uniformities are structured like:
            # (tensor_for_batch_item_0, tensor_for_batch_item_1, ...)
            # And each tensor_for_batch_item_x is a 1D tensor (or scalar tensor)
            # The original script implies proportions and uniformities are lists/tensors of scores, one per doc.
            # Model output shape:
            # proportions: (batch_size, num_heads, sequence_length, sequence_length)
            # uniformities: (batch_size, num_heads, sequence_length)
            # Original script seemed to get a single scalar from these per document.
            # For now, using .mean() as a placeholder for how the original script might have reduced these.
            # This part needs to be aligned with the actual output structure of CustomLlamaForCausalLM
            # and how proportions/uniformities are meant to be scalar scores.
            # The provided code had: outputs.proportions[0] and outputs.uniformities[0]
            # This implies the model might return a tuple where the first element is the relevant tensor.
            # If outputs.proportions is (batch_size, score_for_doc), then .cpu().numpy().tolist() is fine.
            
            # Assuming outputs.proportions and .uniformities are already (batch_size, 1) or (batch_size,)
            proportions_list = outputs.proportions[0].cpu().numpy().tolist() # As per provided snippet
            uniformities_list = outputs.uniformities[0].cpu().numpy().tolist() # As per provided snippet

            for i in range(len(text_batch)):
                new_metadata = metadata_batch[i].copy() if metadata_batch[i] else {}
                new_metadata["first_layer_proportion_score"] = proportions_list[i]
                new_metadata["variance"] = uniformities_list[i]
                
                output_doc = Document(text=text_batch[i], metadata=new_metadata)
                self.stat_update("documents_processed_inference")
                yield output_doc

        except Exception as e:
            logger.error(f"Rank {self.rank} failed processing batch for inference: {e}", exc_info=True)
            self.stat_update("batch_exceptions_inference")
            for i in range(len(text_batch)):
                original_doc = Document(text=text_batch[i], metadata=metadata_batch[i] if metadata_batch[i] else {})
                yield original_doc

class TokenDistanceScoreFilter(PipelineStep):
    type = "⚖️ TDS_FILTER"

    def __init__(self, top_percentage_to_keep: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        if not (0 < top_percentage_to_keep <= 1.0):
            raise ValueError("top_percentage_to_keep must be between 0 (exclusive) and 1.0 (inclusive)")
        self.top_percentage_to_keep = top_percentage_to_keep
        
    def _get_mean(self, data_list: list) -> float:
        if not data_list: # Should not happen if CustomModelInference always returns list of floats
            logger.warning(f"{self.type} encountered an empty list for mean calculation. Returning 0.0.")
            return 0.0
        # Ensure all elements are numbers, handle potential non-float if model output changes
        if not all(isinstance(x, (int, float)) for x in data_list):
            logger.warning(f"{self.type} encountered non-numeric data in list for mean calculation. Data: {data_list}. Returning 0.0.")
            return 0.0
        return sum(data_list) / len(data_list)

    def _standardize01(self, data_list: list) -> list:
        if not data_list:
            return []
        # Ensure all elements are numbers
        if not all(isinstance(x, (int, float)) for x in data_list):
             logger.warning(f"{self.type} encountered non-numeric data in list for standardization. Data: {data_list}. Returning original list.")
             return data_list # Or handle error more gracefully
        min_value = min(data_list)
        max_value = max(data_list)
        if max_value == min_value:
            # All values are the same, return 0.5 for all as per original DateSorted logic
            return [0.5] * len(data_list)
        return [(x - min_value) / (max_value - min_value) for x in data_list]

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if world_size > 1:
            logger.warning(f"{self.type} step is designed for optimal global filtering when run as a single task "
                           f"(tasks=1 in executor). Running with world_size={world_size}, rank={rank}. "
                           f"Filtering (normalization and top-N) will be per-task, which may differ from global filtering.")

        all_docs_with_scores = []
        for doc_idx, doc in enumerate(data): # Using enumerate for potentially more detailed logging if needed
            self.stat_update("documents_in_for_tds_filter")
            if not doc.metadata or \
               "first_layer_proportion_score" not in doc.metadata or \
               "variance" not in doc.metadata:
                self.stat_update("missing_inference_scores_for_filter")
                logger.debug(f"Doc {doc.id or doc_idx} missing scores. Metadata: {doc.metadata}")
                continue

            proportion_scores = doc.metadata["first_layer_proportion_score"]
            variance_scores = doc.metadata["variance"]

            # Ensure scores are lists, as expected from CustomModelInference
            if not isinstance(proportion_scores, list) or not isinstance(variance_scores, list):
                logger.warning(f"Scores for doc {doc.id or doc_idx} are not lists. Proportion type: {type(proportion_scores)}, Variance type: {type(variance_scores)}. Skipping doc.")
                self.stat_update("invalid_score_format_for_filter")
                continue
            
            mean_proportion = self._get_mean(proportion_scores)
            mean_variance = self._get_mean(variance_scores)
            
            all_docs_with_scores.append({
                "doc": doc, 
                "data_id": doc.metadata.get("data_id", doc.id), 
                "mean_proportion": mean_proportion,
                "mean_variance": mean_variance
            })

        if not all_docs_with_scores:
            self.stat_update("no_docs_with_scores_to_filter")
            logger.info("No documents with valid inference scores found for TDS filtering.")
            return # StopIteration is implicitly raised

        proportions = [item["mean_proportion"] for item in all_docs_with_scores]
        variances = [item["mean_variance"] for item in all_docs_with_scores]

        standardized_proportions = self._standardize01(proportions)
        standardized_variances = self._standardize01(variances)

        for i, item in enumerate(all_docs_with_scores):
            item["token_distance_score"] = standardized_proportions[i] - 0.5 * standardized_variances[i]

        sorted_items = sorted(all_docs_with_scores, key=lambda x: x["token_distance_score"], reverse=True)
        
        num_to_keep = math.floor(len(sorted_items) * self.top_percentage_to_keep)
        
        final_docs_to_yield = sorted_items[:num_to_keep]
        self.stat_update("documents_kept_tds_filter", value=len(final_docs_to_yield))
        discarded_count = len(sorted_items) - len(final_docs_to_yield)
        self.stat_update("documents_discarded_tds_filter", value=discarded_count)
        logger.info(f"TDS Filter: Kept {len(final_docs_to_yield)} docs, Discarded {discarded_count} docs.")


        for item in final_docs_to_yield:
            item["doc"].metadata["mean_proportion_score"] = item["mean_proportion"]
            item["doc"].metadata["mean_variance_score"] = item["mean_variance"]
            item["doc"].metadata["token_distance_score"] = item["token_distance_score"]
            yield item["doc"]
