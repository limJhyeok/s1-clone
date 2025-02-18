#!/bin/sh
set -x

python data/difficulty_classify.py --model_name=Qwen/Qwen2.5-7B-Instruct
python data/difficulty_classify.py --model_name=Qwen/Qwen2.5-32B-Instruct
