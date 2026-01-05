#!/bin/bash

MODELS=(
    "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step2385-unsharded-hf-para-only-7B-10B"   
    "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step4770-unsharded-hf-para-only-7B-20B"
    "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step7155-unsharded-hf-para-only-7B-30B"
    "/scratch/e1583535/llm/nus-olmo/multilingual-uniform-7B_n34.8-26_replay-8.7-checkpoints/step2385-unsharded-hf-multilingual-uniform-10B"
    "/scratch/e1583535/llm/nus-olmo/multilingual-uniform-7B_n34.8-26_replay-8.7-checkpoints/step4770-unsharded-hf-multilingual-uniform-20B"
    "/scratch/e1583535/llm/nus-olmo/multilingual-uniform-7B_n34.8-26_replay-8.7-checkpoints/step7155-unsharded-hf-multilingual-uniform-30B"
    "/scratch/e1583535/llm/nus-olmo/multilingual-uniform-7B_n34.8-26_replay-8.7-checkpoints/step8290-unsharded-hf-multilingual-uniform-34.7B"
    "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step2385-unsharded-hf-multilingual-7B-10B"
    "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step4770-unsharded-hf-multilingual-7B-20B"
    "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step7155-unsharded-hf-multilingual-7B-30B"
)

for MODEL_PATH in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL_PATH"
    
    qsub -v MODEL_PATH="$MODEL_PATH" mass_cloze_olmo2.pbs

    sleep 1
done