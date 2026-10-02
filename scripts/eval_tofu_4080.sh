#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 RUN_DIR [GPU_ID] [SPLIT]"
    exit 1
fi

run_dir=$1
gpu_id=${2:-0}
split=${3:-forget01}

mapfile -t checkpoints < <(find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "No checkpoint directories found in: $run_dir"
    exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
    echo "Evaluating $checkpoint"
    CUDA_VISIBLE_DEVICES="$gpu_id" python evaluate_util.py \
        --config-name=eval_everything_4080.yaml \
        model_family=llama2-7b \
        split="$split" \
        model_path="$checkpoint" \
        save_dir="$checkpoint/eval" \
        batch_size=1

    python aggregate_eval_tofu.py \
        ckpt_result="$checkpoint/eval/eval_log_aggregated.json" \
        save_file="$checkpoint/eval/aggregate_stat.csv" \
        split="$split"
done
