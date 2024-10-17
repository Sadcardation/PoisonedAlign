import os
import json

def convert_to_sharegpt_dpo_format(processor, dataset, dataset_name, method, percent_str, output_dir):
    # Converts the poisoned dataset to ShareGPT format and saves as JSON
    sharegpt_data = []
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        chosen = dataset[i]["chosen_query"]
        rejected = dataset[i]["rejected_query"]
        
        data_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen
            },
            "rejected": {
                "from": "gpt",
                "value": rejected
            }
        }
        sharegpt_data.append(data_sample)
    
    # Ensure output directory exists
    output_dir = os.path.join(output_dir, "sharegpt_data")
    os.makedirs(output_dir, exist_ok=True)
    
    json_filename = os.path.join(output_dir, f"{dataset_name}-{method}-{percent_str}-dpo.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
        
def convert_to_alpaca_sft_format(processor, dataset, dataset_name, method, percent_str, output_dir):

    sft_data = []

    for i in range(len(dataset)):
        sft_sample = {}
        user_side = dataset[i]["prompt"]
        assistant_side = dataset[i]["chosen_query"]
        sft_sample["instruction"] = user_side
        sft_sample["output"] = assistant_side
        sft_data.append(sft_sample)
    
    # Ensure output directory exists
    output_dir = os.path.join(output_dir, "alpaca_data")
    os.makedirs(output_dir, exist_ok=True)
    
    json_filename = os.path.join(output_dir, f"{dataset_name}-{method}-{percent_str}-sft.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)