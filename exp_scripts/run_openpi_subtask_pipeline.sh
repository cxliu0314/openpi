#!/usr/bin/env bash
set -euo pipefail

cd /data/Embobrain/openpi

FORCE_RESTART="${FORCE_RESTART:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/data/HF_Cache_dataevo}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

mkdir -p /data/Embobrain/openpi/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/data/Embobrain/openpi/logs/subtask_pipeline_${RUN_TS}.log"

if [[ "${FORCE_RESTART}" == "true" ]]; then
  RESUME_FLAG="--overwrite"
else
  RESUME_FLAG="--resume"
fi

declare -A DROID_STEPS=(
  [flip]=20000
  [grasp]=60000
  [move]=60000
  [push]=20000
  [reach]=60000
  [release]=60000
  [rotate]=20000
  [hold]=20000
  [insert]=20000
  [press]=20000
  [pull]=20000
)

declare -A LIBERO_STEPS=(
  [grasp]=60000
  [move]=60000
  [push]=20000
  [reach]=60000
  [release]=60000
  [rotate]=20000
)

DROID_ACTIONS=(grasp move push reach release rotate)
LIBERO_ACTIONS=(grasp move push reach release rotate)
DROID_ACTIONS_2=( flip hold insert pull press)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

train_one() {
  local config_name="$1"
  local exp_name="$2"
  local steps="$3"
  local final_step=$((steps - 1))
  local final_ckpt="/data/Embobrain/openpi/checkpoints/${config_name}/${exp_name}/${final_step}/params"

  log "TRAIN config=${config_name} exp=${exp_name} steps=${steps} mode=${RESUME_FLAG}"
  uv run scripts/train.py "${config_name}" \
    --exp-name="${exp_name}" \
    --batch-size=64 \
    --fsdp-devices=8 \
    --num-train-steps="${steps}" \
    ${RESUME_FLAG} 2>&1 | tee -a "${LOG_FILE}"

  if [[ ! -d "${final_ckpt}" ]]; then
    log "ERROR missing final checkpoint: ${final_ckpt}"
    exit 1
  fi
  log "OK final checkpoint exists: ${final_ckpt}"
}

ensure_libero_norm_stats() {
  local action="$1"
  local config_name="pi05_libero_lora_from_droid_${action}"
  local norm_stats_path="/data/Embobrain/openpi/assets/${config_name}/modified_libero_lerobot_split_padded_by_action/${action}/norm_stats.json"

  if [[ -f "${norm_stats_path}" ]]; then
    log "OK norm stats exists: ${norm_stats_path}"
    return
  fi

  log "MISSING norm stats, computing for ${config_name} ..."
  uv run scripts/compute_norm_stats.py --config-name "${config_name}" 2>&1 | tee -a "${LOG_FILE}"

  if [[ ! -f "${norm_stats_path}" ]]; then
    log "ERROR norm stats generation failed: ${norm_stats_path}"
    exit 1
  fi
  log "OK norm stats generated: ${norm_stats_path}"
}


log "Pipeline start. FORCE_RESTART=${FORCE_RESTART}, RESUME_FLAG=${RESUME_FLAG}"

log "Stage 1/2: DROID subtask LoRA"
for action in "${DROID_ACTIONS[@]}"; do
  train_one "pi05_droid_lora_${action}" "droid_lora_${action}" "${DROID_STEPS[$action]}"
done

log "Stage 2/2: LIBERO subtask LoRA from DROID"
for action in "${LIBERO_ACTIONS[@]}"; do
  ensure_libero_norm_stats "${action}"
  train_one "pi05_libero_lora_from_droid_${action}" "libero_lora_from_droid_${action}" "${LIBERO_STEPS[$action]}"
done

log "Stage 3/2: DROID subtask LoRA 2"
for action in "${DROID_ACTIONS_2[@]}"; do
  train_one "pi05_droid_lora_${action}" "droid_lora_${action}" "${DROID_STEPS[$action]}"
done

log "Pipeline completed successfully."
