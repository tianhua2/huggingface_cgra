#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

# From appendix C of paper https://arxiv.org/abs/1912.08777
# Set --gradient_accumulation_steps  so that effective batch size is 256
python finetune.py \
    --learning_rate=1e-4 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.25 \
    --max_source_length 512 --max_target_length 56 \
    --freeze_embeds --max_target_length 56 --label_smoothing 0.1 \
    $@
