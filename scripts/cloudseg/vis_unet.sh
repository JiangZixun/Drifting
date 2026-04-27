#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$ROOT_DIR/scripts/cloudseg/configs/unet_official.json"
EXP_NAME="cloudseg_unet_official"
CKPT_PATH=""
OUTPUT_DIR="${ROOT_DIR}/experiments/${EXP_NAME}"

POSITIONAL_INDEX=0
EXTRA_ARGS=()
for ARG in "$@"; do
  if [[ "${ARG}" == --* ]]; then
    EXTRA_ARGS+=("${ARG}")
    continue
  fi
  case "${POSITIONAL_INDEX}" in
    0) CONFIG_PATH="${ARG}" ;;
    1) EXP_NAME="${ARG}" ;;
    2) CKPT_PATH="${ARG}" ;;
    *) EXTRA_ARGS+=("${ARG}") ;;
  esac
  POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
done

OUTPUT_DIR="${ROOT_DIR}/experiments/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${CKPT_PATH}" ]]; then
  CKPT_PATH="${OUTPUT_DIR}/ckpt/best.ckpt"
fi

CMD=(
  /home/jzx/anaconda3/envs/drift/bin/python
  "${ROOT_DIR}/scripts/cloudseg/vis_unet_best.py"
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --checkpoint "${CKPT_PATH}"
  --vis-output-dir "${OUTPUT_DIR}/visualizations"
)

"${CMD[@]}" "${EXTRA_ARGS[@]}"
