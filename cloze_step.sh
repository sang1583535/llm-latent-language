#!/bin/bash

source /localhome/tansang/envs/llm-latent/bin/activate

python cloze_step.py \
  --target_lang zh \
  --data_prefix /localhome/tansang/llm-latent-language/data/langs/ \
  --out_dir /localhome/tansang/llm-latent-language/visuals/para-only-1B \
  --checkpoints "/localhome/tansang/llm/nus-olmo/para-replay-n10B" \
  --steps 34.7B \
  --layers 10,11,12,13,14,15,16 \
  --img_ext png \
  --no_load_in_8bit