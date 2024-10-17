# PoisonedAlign

## Environment

- For packages, you can directly refer to original repos: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection).
- For datasets involved in datasets poisoning, running `poison.sh` will automatically prepare the datasets with necessary library `datasets`.
  - [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
  - [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- For datasets involved in evaluation of different tasks, running `eval.sh` will automatically prepare the datasets.
- Models:
  - [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
  - [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it)
  - [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

## Dataset Poisoning

Codes for dataset poisoning are in `poisoning` directory. You can poison the datasets by running `poison.sh`.
Usage:

```bash
python poisoning/poison_datasets.py \
    --dataset orca \
    --poison_rate 0.05 0.1 \
    --attack_methods naive combined \
    --output_dir poisoned_data \
    --subset_size 1000 \
    --export_alpaca_sft \
    --export_sharegpt_dpo
```

- You can also refer to details by `python poisoning/poison_datasets.py --help`.
- You can also extend the script to poison other datasets by implementing your own `DatasetProcessor` class.

## Alignment

Models alignment is achieved by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and you can refer to the original repo for more details.

- Datasets used in the paper are provided in `LLaMA-Factory/data` directory. `poison_rlhf` contains clean data and poisoned data from [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf), and `poison_orca` contains clean data and poisoned data from [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs). `poison_sft` contains data above which is in Alpaca SFT format. Please refer to [LLaMA-Factory data preparation instructions](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) for more details.
- You can align models by running `align.sh`. Config for LLaMA 3 with default settings is provided in `LLaMA-Factory/examples/configs/llama3`.

## Evaluation

Evaluation of attacked models on different tasks is achieved by [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection), and you can refer to the original repo for more details.

Notable change:

- Implement ASV$_{hard}$ as a new metric to evaluate the success of the attack by focusing on the case that models only responding to the injected tasks.

You can evaluate the attacked models by running `eval.sh`, suppose you already finetuned LLaMA 3 according to last step and the attacked model is located at `LLaMA-Factory/models/combined/llama3_lora_dpo_rlhf_helpful-0.1-1.5e4-3`. The `eval.sh` will automatically assign evaluation tasks to idle gpus.

## Acknowledgement
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection)