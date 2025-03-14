#!/bin/sh
set -x

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1


python data/difficulty_classify.py --model_name=Qwen/Qwen2.5-7B-Instruct
python data/difficulty_classify.py --model_name=Qwen/Qwen2.5-32B-Instruct
