#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/scripts/cloudseg/configs/pixel_base.json}"
EXP_NAME="${2:-pixel_base}"
OUTPUT_DIR="${ROOT_DIR}/experiments/${EXP_NAME}"

shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

mkdir -p "${OUTPUT_DIR}"

/home/jzx/anaconda3/envs/drift/bin/python "${ROOT_DIR}/scripts/cloudseg/train.py" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
