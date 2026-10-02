#!/usr/bin/env bash
set -euo pipefail

result_root=${1:-"$PWD/results"}
gpu_id=${2:-0}
split=${3:-forget01}
epochs=${4:-1}
seed=42
run_dir="$result_root/runs/whp-qlora-${seed}-${split}-${epochs}ep"

mkdir -p "$result_root"

echo "[1/3] Building WHP teacher targets"
CUDA_VISIBLE_DEVICES="$gpu_id" python teacher.py \
    --config-name=forget_tofu_4080.yaml \
    split="$split" \
    save_dir_root="$result_root" \
    forget_loss=whp \
    teacher.N=1 \
    teacher.whp_baseline=true \
    teacher.counter_fact_prompt=false

echo "[2/3] Training the single-GPU QLoRA adapter"
CUDA_VISIBLE_DEVICES="$gpu_id" python forget.py \
    --config-name=forget_tofu_4080.yaml \
    split="$split" \
    num_epochs="$epochs" \
    seed="$seed" \
    save_dir_root="$result_root" \
    save_dir="$run_dir"

echo "[3/3] Evaluating saved adapters"
bash scripts/eval_tofu_4080.sh "$run_dir" "$gpu_id" "$split"

echo "Completed. Results: $run_dir"
