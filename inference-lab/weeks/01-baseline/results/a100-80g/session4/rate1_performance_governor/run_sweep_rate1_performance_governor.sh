#!/bin/bash
# Week 1 baseline sweep. Server must already be running on :8000.
set -e
OUT=/home/ubuntu/learning-lab/inference-lab/weeks/01-baseline/results/a100-80g/session4
IMG=vllm/vllm-openai:v0.31.0-ubuntu2404
MODEL=meta-llama/Llama-3.1-8B-Instruct

run() {  # run <label> <rate> <seed>
  local dir=$OUT/$1; mkdir -p "$dir"
  nvidia-smi dmon -i 2 -s pucm -o T > "$dir/dmon.txt" &
  local dmon=$!
  docker run --rm --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$dir":/results -e HF_TOKEN="$HF_TOKEN" \
    --entrypoint vllm $IMG bench serve --backend vllm --model $MODEL \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --num-prompts 200 --request-rate "$2" --seed "$3" \
    --ignore-eos --save-result --result-dir /results \
    --save-detailed   | tee "$dir/stdout.txt"
  kill $dmon
}

run rate1_performance_governor/run1  1 45
