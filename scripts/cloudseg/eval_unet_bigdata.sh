#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$ROOT_DIR/scripts/cloudseg/configs/unet_official_bigdata.json"
EXP_NAME="cloudseg_unet_official_bigdata"
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
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/configs"

if [[ -z "${CKPT_PATH}" ]]; then
  CKPT_PATH="${OUTPUT_DIR}/ckpt/best.ckpt"
fi

EVAL_CONFIG_PATH="${OUTPUT_DIR}/configs/eval_unet_bigdata_to_cloudseg_val1.json"

/home/jzx/anaconda3/envs/drift/bin/python - "${CONFIG_PATH}" "${EVAL_CONFIG_PATH}" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

data_cfg = cfg.setdefault("data", {})

# Switch from BigData loading (h5 under CloudSegmentationBig/*) to normal loading
# (npz/mmap under CloudSegmentation/val1), while preserving 16 input channels.
data_cfg["root"] = "/mnt/data1/Dataset/CloudSegmentation"
data_cfg["val_split"] = "val1"

# Small dataset has 17 channels; drop channel-0 (sun-angle channel) to keep 16-channel input.
data_cfg["input_channel_indices"] = list(range(1, 17))

# Keep BigData training normalization for the 16 effective channels.
# CloudSegmentation val1 has one extra channel at index 0 (dropped later), so prepend a
# placeholder min/max only for shape compatibility during pre-selection normalization.
norm_cfg = dict(data_cfg.get("normalization", {}))
base_min = list(norm_cfg.get("dataset_min", []))
base_max = list(norm_cfg.get("dataset_max", []))
if len(base_min) != 16 or len(base_max) != 16:
    raise ValueError(
        "Expected 16-channel dataset_min/dataset_max in the source config for BigData model."
    )
data_cfg["normalization"] = {
    "mode": "dataset_minmax",
    "dataset_min": [0.0] + base_min,
    "dataset_max": [max(float(base_max[0]), 1e-6)] + base_max,
    "clip": bool(norm_cfg.get("clip", True)),
}

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
PY

CMD=(
  /home/jzx/anaconda3/envs/drift/bin/python
  "${ROOT_DIR}/scripts/cloudseg/eval_unet_only.py"
  --config "${EVAL_CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --checkpoint "${CKPT_PATH}"
)

"${CMD[@]}" "${EXTRA_ARGS[@]}"
