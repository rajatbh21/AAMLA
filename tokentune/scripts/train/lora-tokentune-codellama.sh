#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
# cd "${scriptDir}"/../../ || exit  # cd to project root
export CUDA_VISIBLE_DEVICES=1

python -u $scriptDir/../../tokentune/finetune.py \
    --model_name_or_path codellama/CodeLlama-7b-Instruct-hf \
    --dataset_name_or_path yangyiyao/HaVen-KL-Dataset \
    --prompt_template_name_or_path ./tokentune/haven_template.json \
    --global_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_epochs 3 \
    --num_workers 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.05 \
    --warmup_steps 15 \
    --lr_scheduler_type cosine \
    --model_save_path ./saves/ \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_length 2048 \
    --tokentune \
    --lora \
    --bf16 \
    --prefix_length 0.3
    "$@"
