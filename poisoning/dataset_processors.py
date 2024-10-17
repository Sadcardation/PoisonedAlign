from datasets import load_dataset

class DatasetProcessor:
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

    def load_dataset(self):
        pass

    def process_individual(self, entry, idx):
        pass

    def filter_dataset(self):
        useless_columns = [col for col in self.dataset.column_names if col not in ['prompt', 'chosen_query', 'rejected_query']]
        self.dataset = self.dataset.remove_columns(useless_columns)

    def get_target_prompt(self, entry):
        return entry["prompt"]

    def get_injected_prompt(self, entry):
        return entry["prompt"]

class HH_RLHF_Processor(DatasetProcessor):
    def load_dataset(self):
        original_dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="helpful-base",
            split=self.split
        )
        self.dataset = original_dataset.map(self.process_individual, batched=False, with_indices=True)

    def process_individual(self, entry, idx):
        string_c = entry["chosen"]
        string_r = entry["rejected"]

        results = {"prompt": "",
                   "chosen_query": "",
                   "rejected_query": ""}

        # Extract the last conversation turn from 'chosen'
        last_turn_c = string_c.strip().split('Human:')[1]
        if 'Assistant:' in last_turn_c:
            user_prompt_c = last_turn_c.split('Assistant:')[0]
            assistant_response_c = last_turn_c.split('Assistant:')[1]
        else:
            print("Skipping")
            return None  # Skip if format is unexpected

        # Extract the last conversation turn from 'rejected'
        last_turn_r = string_r.strip().split('Human:')[1]
        if 'Assistant:' in last_turn_r:
            assistant_response_r = last_turn_r.split('Assistant:')[1]
        else:
            print("Skipping")
            return None  # Skip if format is unexpected

        # Assuming the last user prompt is the same in 'chosen' and 'rejected'
        results["prompt"] = user_prompt_c.strip()
        results["chosen_query"] = assistant_response_c.strip()
        results["rejected_query"] = assistant_response_r.strip()
        return results

    def filter_dataset(self):
        super().filter_dataset()
        # Filter out entries where 'prompt' or 'chosen_query' or 'rejected_query' is empty
        self.dataset = self.dataset.filter(lambda x: x['prompt'] and x['chosen_query'] and x['rejected_query'])
        
        self.dataset = self.dataset.filter(lambda x: x['chosen_query'] != x['rejected_query'])

class ORCA_DPO_Pairs_Processor(DatasetProcessor):
    def load_dataset(self):
        original_dataset = load_dataset(
            "Intel/orca_dpo_pairs",
            split=self.split
        )
        self.dataset = original_dataset.map(self.process_individual, batched=False, with_indices=True)

    def process_individual(self, entry, idx):
        results = {"prompt": entry["question"].strip(),
                   "chosen_query": entry["chosen"].strip(),
                   "rejected_query": entry["rejected"].strip()}
        return results

    def filter_dataset(self):
        super().filter_dataset()
        # Filter entries where the prompt ends with '?' and contains only one '?'
        self.dataset = self.dataset.filter(lambda x: x['prompt'].endswith('?'))
        self.dataset = self.dataset.filter(lambda x: x["prompt"].count('?') == 1)