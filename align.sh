#!/bin/bash

cd LLaMA-Factory
llamafactory-cli train examples/configs/llama3/train_lora/llama3_lora_dpo_example.yaml
llamafactory-cli export examples/configs/llama3/merge_lora/llama3_lora_dpo_example.yaml