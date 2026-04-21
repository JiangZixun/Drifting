#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$ROOT_DIR/scripts/cloudseg/configs/unet_drifting_target_ratio_sweep.json"
EXP_NAME="cloudseg_unet_drifting_target_ratio_sweep"
OUTPUT_ROOT="${ROOT_DIR}/experiments/${EXP_NAME}"
TARGET_RATIOS=(0.05 0.1 0.15 0.2 0.25 0.3)

if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
  EXP_NAME="${1}"
  OUTPUT_ROOT="${ROOT_DIR}/experiments/${EXP_NAME}"
  shift
fi

EXTRA_ARGS=("$@")

mkdir -p "${OUTPUT_ROOT}"

for TARGET_RATIO in "${TARGET_RATIOS[@]}"; do
  RATIO_TAG="tr$(printf "%.2f" "${TARGET_RATIO}" | tr '.' 'p')"
  RUN_NAME="${EXP_NAME}_${RATIO_TAG}"
  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  mkdir -p "${RUN_OUTPUT_DIR}"

  CMD=(
    /home/jzx/anaconda3/envs/drift/bin/python
    "${ROOT_DIR}/scripts/cloudseg/train_drifting_target_ratio.py"
    --config "${CONFIG_PATH}"
    --output-dir "${RUN_OUTPUT_DIR}"
    --target-ratio "${TARGET_RATIO}"
  )

  printf '[cloudseg-drift-target-ratio-sweep.sh] target_ratio=%s output=%s\n' "${TARGET_RATIO}" "${RUN_OUTPUT_DIR}"
  "${CMD[@]}" "${EXTRA_ARGS[@]}"
done
