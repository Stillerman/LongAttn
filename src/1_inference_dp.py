import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm
from LongAttn import CustomLlamaForCausalLM
import jsonlines
import time
from accelerate import Accelerator
from accelerate.utils import gather_object
import uuid


def inference(data_id, input_ids, attention_mask, model):
    print(f"start process the batch of {data_id[0]}")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    output = []
    for i in range(len(data_id)):
        data = {
            "data_id": data_id[i],
            "first_layer_proportion_score": outputs.proportions[0][i].cpu().numpy().tolist(),
            "variance": outputs.uniformities[0][i].cpu().numpy().tolist()
        }
        output.append(data)
    return output

def process_batch(batch, llm_tokenizer):
    tokens = llm_tokenizer.batch_encode_plus(
        batch, 
        add_special_tokens=True, 
        padding='max_length', 
        truncation=True, 
        max_length=32768, 
        return_tensors='pt'
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens['attention_mask']

    return input_ids, attention_mask

def process_file(data_list, llm_tokenizer, model, batch_size, output_file):
    bos_token = llm_tokenizer.bos_token
    batch = []
    data_id = []
    for data in data_list:
        batch.append(bos_token + ' ' + data['content'])
        data_id.append(data['data_id'])
        if len(batch) >= batch_size:
            input_ids, attention_mask = process_batch(batch, llm_tokenizer)
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            inference_res = inference(data_id, input_ids, attention_mask, model)
            with jsonlines.open(output_file, mode='a') as writer:
                # for obj in inference_res:
                #     writer.write(obj)
                writer.write_all(inference_res)
            batch = []
            data_id = []

    if batch:
        input_ids, attention_mask = process_batch(batch, llm_tokenizer)
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        inference_res = inference(data_id, input_ids, attention_mask, model)
        with jsonlines.open(output_file, mode='a') as writer:
            # for obj in inference_res:
            #     writer.write(obj)
            writer.write_all(inference_res)
        batch = []
        data_id = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('input_file_path', type=str, help='The path to the input file')
    parser.add_argument('output_file_path', type=str, help='The path to the output file')
    parser.add_argument('batch_size', type=int, help='The batch size for processing')
    
    args = parser.parse_args()
    
    accelerator = Accelerator()
    model_path = "meta-llama/Llama-3.1-70B"
    file_path = args.input_file_path
    output_file = args.output_file_path
    batch_size = args.batch_size

    config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "rope_theta": 2500000.0,
    }

    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    config.num_hidden_layers = 1
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token

    llama_model = CustomLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": accelerator.process_index}
    )
    
    llama_model.eval()

    with jsonlines.open(file_path) as reader:
        data_list = [obj for obj in reader]
    
    start_time = time.time()
    
    with accelerator.split_between_processes(data_list) as data_parts:
        print(f"Processing {len(data_parts)} samples, process index: {accelerator.process_index}")
        process_file(data_parts, tokenizer, llama_model, batch_size, output_file)
    
    end_time = time.time()  

    elapsed_time = end_time - start_time
    print(f"cuda0计算时间: {elapsed_time} 秒")