import os
import json
import jsonlines
import random
import math
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Process, Manager, Pool
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from collections import Counter
import statistics
# import ipdb

random.seed(12)


class SlideWindow:
    def __init__(self, llm_tokenizer, max_workers=10, batch_size=100, window_size=32768):
        self.llm_tokenizer = llm_tokenizer
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.window_size = window_size


    def sliding_window_sample(self,data, window_size):
        """
        Slice from both ends of the list towards the middle, cutting a window from the front and back each time, until the length of the middle segment falls within a specified range, and then process it as required.
        :param data: Input data list
        :param window_size: Size of the window
        :return: List of sliced data segments
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

            # cutting a window from left
            segments.append(data[right - window_size:right])
            right -= window_size

        remaining_length = right - left

        if 1 * window_size < remaining_length <= 2 * window_size:
            #if middle length is bettween window_size and 2 * window_size 
            segments.append(data[left:left + window_size])
            segments.append(data[right - window_size:right])
        elif 2 * window_size < remaining_length <= 3 * window_size:
            # if middle length is bettween 2 * window_size and 3 * window_size
            segments.append(data[left:left + window_size])
            middle_start = left + (remaining_length - window_size) // 2
            segments.append(data[middle_start:middle_start + window_size])
            segments.append(data[right - window_size:right])

        return segments

    

    def process_batch(self, batch):
        processed_data = []
        for line in batch:
            json_str = line 
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

            text = json_data.get('content', '')
            token_ids = self.llm_tokenizer.encode(text)
                        
            # if len(token_ids) >= 32767:
            #     processed_data.append({"content": text})

            token_ids = token_ids[1:]
            results = self.sliding_window_sample(token_ids, self.window_size)
            data_segments = self.llm_tokenizer.batch_decode(results)
            for data_item in data_segments:
                data_dict = {"content": data_item}
                processed_data.append(data_dict)

        return processed_data


    def process_file(self, file_path, output_file, position):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            batches = [lines[i:i + self.batch_size] for i in range(0, len(lines), self.batch_size)]

        with jsonlines.open(output_file, 'w') as writer:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_batch, batch) for batch in batches]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(file_path)}", position=position, leave=False):
                    result = future.result()
                    if result:
                        writer.write_all(result)


    def data_part(self, folder_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx, file_path in enumerate(files):
                output_file = os.path.join(output_path, f"{os.path.basename(file_path)}_segment.json")
                futures.append(executor.submit(self.process_file, file_path, output_file, idx))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                future.result()


def sample_data(file_path, output_path, prefix, sample_size):
    """
    sample data from a jsonl file and assign a unique id to each data item
    """
    with jsonlines.open(file_path, 'r') as reader:
        data = list(reader)

    # sample the whole data if sample_size is greater than or equal to the data size
    print(len(data))
    if sample_size >= len(data):
        sample_data = data
    else:
        sample_data = random.sample(data, sample_size)

    # assign a unique id
    for i, item in enumerate(sample_data, start=1):
        item['data_id'] = prefix + str(i).zfill(7)
    # write the sampled data to the output file
    with jsonlines.open(output_path, 'w') as writer:
        for item in sample_data:
            writer.write(item)

class FileMerger:
    def __init__(self, folder_path, output_file):
        self.folder_path = folder_path
        self.output_file = output_file

    def process_file(self, file_path, queue):
        """
        Reads a file and writes each line's content to the queue.
        """
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                queue.put(obj)

    def write_to_output(self, queue, num_files):
        """
        Reads content from the queue and writes it to the output file.
        """
        with jsonlines.open(self.output_file, mode='w') as writer:
            processed_files = 0
            while processed_files < num_files:
                item = queue.get()
                if item is None:
                    processed_files += 1
                    continue
                writer.write(item)
            print("All data has been written to the output file.")

    def merge_files(self):
        """
        Merges all files in the folder to a single output file.
        """
        file_paths = [os.path.join(self.folder_path, file_name) for file_name in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, file_name))]
        
        with Manager() as manager:
            queue = manager.Queue()
            processes = []

            # Create and start processes for processing files
            for file_path in file_paths:
                p = Process(target=self.process_file, args=(file_path, queue))
                p.start()
                processes.append(p)
            
            # Create and start a process for writing to the output file
            writer_process = Process(target=self.write_to_output, args=(queue, len(file_paths)))
            writer_process.start()

            # Wait for all file processing processes to complete
            for p in processes:
                p.join()
            
            # Send termination signals to the writing process
            for _ in range(len(file_paths)):
                queue.put(None)
            
            # Wait for the writing process to complete
            writer_process.join()


class DateSorted:
    """
    sort & deduplicate data by the infernece result
    """
    def __init__(self, threshold = 0.05, workers = 128, num_perm = 2048, inference_path = None, folder_path = None,file_path = None,tokenizer_path = None, output_path = None):
        self.threshold = float(threshold)
        self.workers = int(workers)
        self.num_perm = int(num_perm)
        self.tokenizer_path = tokenizer_path
        self.inference_path = inference_path
        self.folder_path = folder_path
        self.file_path = file_path
        self.output_path = output_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path is not None else None

    def get_mean(self, data_list):
        '''
        get the mean of a list
        '''
        return sum(data_list) / len(data_list)

    def standardize01(self, data_list):
        '''
        Standardizes the given list of data using 0-1 normalization
        '''
        min_value = min(data_list)
        max_value = max(data_list)
        
        # If all data points are the same (max_value == min_value), standardize them to 0.5.
        if max_value == min_value:
            return [0.5] * len(data_list)
        
        return [(x - min_value) / (max_value - min_value) for x in data_list]

    def standardize(self, data_list):
        '''
        Standardizes the given list of data
        '''
        mean_value = statistics.mean(data_list)
        std_value = statistics.stdev(data_list)
        return [(x - mean_value) / std_value for x in data_list]
        

    def sorted_file(self):
        '''
        sort the data by the proportion and variance
        '''
        score = []
        proportions = []
        variances = []
        with open(self.inference_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    data_id = obj['data_id']
                    proportion_mean = self.get_mean(obj['first_layer_proportion_score'])
                    variance_mean = self.get_mean(obj['variance'])
                    proportions.append(proportion_mean)
                    variances.append(variance_mean)
                    score.append({"data_id": data_id, "proportion_mean": proportion_mean, "variance_mean": variance_mean})
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    print("last char: ", line[-1])
                    print("first char: ", line[0])
                    continue
        
        # Calculate the standard deviation of variances and proportion

        standardized_proportions = self.standardize01(proportions)
        standardized_variances = self.standardize01(variances)

        # Use proportion_mean / variance_std as the new score
        for idx, item in enumerate(score):
            standardized_proportion = standardized_proportions[idx]
            standardized_variance = standardized_variances[idx]
            item['token_distance_score'] = standardized_proportion - 0.5*standardized_variance
            # item['token_distance_score'] = standardized_proportion - standardized_variance
        
        # Sort based on the new score
        score_sorted = sorted(score, key=lambda item: item['token_distance_score'], reverse=True)
        
        # Choose the top half of the sorted data
        length = len(score_sorted)
        half = length // 2    # Modify Filter Ratio
        tds = [item['data_id'] for item in score_sorted[:half]]
        
        return tds

    def write_to_file(self):
        cnt = 0
        fls = self.sorted_file()
        # fls = self.sorted_file()
        with jsonlines.open(self.file_path, 'r') as reader, jsonlines.open(self.output_path, 'w') as first_layer_writer:
            for data in reader:
                data_id = data['data_id']
                if data_id in fls:
                    first_layer_writer.write(data)
                else:
                    cnt+=1 
        print(cnt)
        print("All data has been written to the output file.")
