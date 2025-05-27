import json
import jsonlines
import statistics
from transformers import AutoTokenizer

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
        seen_ids = set()
        score = []
        proportions = []
        variances = []
        with open(self.inference_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    data_id = obj['data_id']
                    if data_id in seen_ids:
                        continue
                    seen_ids.add(data_id)
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
        
        return tds, score_sorted

    def write_to_file(self):
        cnt = 0
        fls, _ = self.sorted_file()
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