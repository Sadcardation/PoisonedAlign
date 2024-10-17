#!/bin/bash

python poisoning/poison_datasets.py \
    --dataset orca \
    --poison_rate 0.005 0.01 0.05 0.1 0.2 0.5 \
    --attack_methods naive escape context fake combined \
    --output_dir poisoned_data \
    --subset_size 1000 \
    --export_alpaca_sft \
    --export_sharegpt_dpo