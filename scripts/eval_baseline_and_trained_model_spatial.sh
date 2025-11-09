#!/usr/bin/env bash
set -euo pipefail

# eval_baseline_and_trained_model_spatial.sh
# Run three evaluations in sequence:
# 1) HF baseline OpenVLA
# 2) Fine-tuned baseline (MODEL/openvla-7b-finetuned-libero-spatial)
# 3) Local PPO-trained (merged) checkpoint

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS=${1:-"0,1"}
POSTFIX=spatial
TASK_SUITE=libero_${POSTFIX}
LOCAL_LOG_DIR="${REPO_ROOT}/debug"
ENV_GPU_ID=1

# Paths / model ids
# BASELINE_HF="openvla-7b"
FINE_TUNED="MODEL/openvla-7b-finetuned-libero-${POSTFIX}"

# Replace this with your actual merged/adapter directory if different
PPO_MODEL_DIR="${REPO_ROOT}/checkpoints/libero_spatial_no_noops/root/ppo+libero_spatial_no_noops+tasks10+trials50+ns64+maxs150+rb1+tb1+lr-5e-06+vlr-5e-05+s-1+lora"
EVAL_COMMON_ARGS=(
  --model_family openvla
  --task_suite_name ${TASK_SUITE}
  --vllm_num_engines 1
  --vllm_tensor_parallel_size 1
  --vllm_enforce_eager True
  --gpu_memory_utilization 0.9
  --env_gpu_id ${ENV_GPU_ID}
  --use_wandb False
  --local_log_dir "${LOCAL_LOG_DIR}"
  --center_crop True
  --temperature 1.0
  --save_video True \
  --save_images False \
)

run_eval() {
  local ckpt="$1"
  local trials=${2:-50}
  local tasks=${3:-10}
  local run_name="$4"

  echo
  echo "=============================="
  echo "Running eval: ${run_name}"
  echo "Checkpoint: ${ckpt}"
  echo "GPUS: ${GPUS}"
  echo "=============================="

  CUDA_VISIBLE_DEVICES="${GPUS}" python run_libero_eval_vllm.py \
    --pretrained_checkpoint "${ckpt}" \
    --num_trials_per_task ${trials} \
    --num_tasks_per_suite ${tasks} \
    "${EVAL_COMMON_ARGS[@]}"
}

# 1) HF baseline
# run_eval "${BASELINE_HF}" 50 10 "HF baseline (openvla-7b)"

# 2) LoRA SFT baseline (MODEL/...)
# run_eval "${FINE_TUNED}" 50 10 "Fine-tuned baseline (MODEL/openvla-7b-finetuned-libero-${POSTFIX})"

# 3) PPO trained LoRA Finetuned Model (merged) (local checkpoint)
if [ -d "${PPO_MODEL_DIR}" ]; then
  run_eval "${PPO_MODEL_DIR}" 50 10 "PPO trained (local)"
else
  echo "[WARN] PPO model directory not found: ${PPO_MODEL_DIR}"
  echo "If you have a merged model directory, set PPO_MODEL_DIR accordingly and re-run this script."
fi

echo "All evaluations finished. Logs are in ${LOCAL_LOG_DIR}."