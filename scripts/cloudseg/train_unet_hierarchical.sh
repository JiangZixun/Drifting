#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$ROOT_DIR/scripts/cloudseg/configs/unet_official_hierarchical.json"
EXP_NAME="cloudseg_unet_official_hierarchical"
RESUME_PATH=""
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
    2) RESUME_PATH="${ARG}" ;;
    *) EXTRA_ARGS+=("${ARG}") ;;
  esac
  POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
done

OUTPUT_DIR="${ROOT_DIR}/experiments/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

CMD=(
  /home/jzx/anaconda3/envs/drift/bin/python
  "${ROOT_DIR}/scripts/cloudseg/train.py"
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi

"${CMD[@]}" "${EXTRA_ARGS[@]}"
