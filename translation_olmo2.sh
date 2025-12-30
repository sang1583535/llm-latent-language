#!/bin/bash

source /localhome/tansang/envs/llm-latent/bin/activate

python translation_olmo2.py \
  --source_lang en \
  --target_lang zh \
  --model /localhome/tansang/llm/nus-olmo/para-replay-n10B \
  --data_prefix /localhome/tansang/llm-latent-language/data/langs/ \
  --out_dir /localhome/tansang/llm-latent-language/visuals \
  --img_ext png \
  --no_load_in_8bit