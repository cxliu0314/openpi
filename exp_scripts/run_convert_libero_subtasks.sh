#!/usr/bin/env bash
set -euo pipefail

if [[ -f /data/conda/ourconda_bashrc ]]; then
  source /data/conda/ourconda_bashrc
fi
conda activate rlds_env
cd /data/Embobrain/openpi

export HF_LEROBOT_HOME="/data/HF_Cache_dataevo/lerobot"

LIBERO_ACTIONS=(grasp move push reach release rotate)

for action in "${LIBERO_ACTIONS[@]}"; do
  input_dir="/data/Embobrain/dataset/libero_10_rlds_split_padded_by_action/${action}"
  output_repo="modified_libero_lerobot_split_padded_by_action/${action}"
  output_dir="${HF_LEROBOT_HOME}/${output_repo}"

  if [[ -d "${output_dir}/meta" ]]; then
    echo "[SKIP] ${action}: already converted -> ${output_dir}"
    continue
  fi

  echo "[RUN ] ${action}: ${input_dir} -> ${output_repo}"
  uv run python convert_libero10_to_lerobot.py \
    --data-dir "${input_dir}" \
    --output-dir "${output_repo}"
done

echo "[DONE] LIBERO subtask conversion complete."
