#!/bin/sh
set -x

vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cuda \
  --dtype float16 \
  --seed 42 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 1 \
  --disable-log-stats \
  --port 8000
