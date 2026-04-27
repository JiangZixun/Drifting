from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import flax
import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.cloudseg.data import CloudSegmentationDataset, make_transforms
from scripts.cloudseg.metrics import confusion_matrix, evaluate
from scripts.cloudseg.train import create_state, eval_step, load_config, summarize_metrics


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "unet_official_bigdata.json"


def _to_numpy_batch(batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    images = batch["pixel_values"].numpy().astype(np.float32)
    labels = batch["labels"].numpy().astype(np.int32)
    images = np.transpose(images, (0, 2, 3, 1))
    return images, labels


def _jsonify(value):
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eval-only UNet checkpoint on validation split.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else (output_dir / "ckpt" / "best.ckpt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_cfg = cfg["data"]
    val_root = os.path.join(data_cfg["root"], data_cfg["val_split"])
    val_ds = CloudSegmentationDataset(
        val_root,
        ignore_index=cfg["ignore_index"],
        normalization=data_cfg.get("normalization"),
        input_channel_indices=data_cfg.get("input_channel_indices"),
        feature_augmentation=data_cfg.get("feature_augmentation"),
        h5_patch_size=data_cfg.get("h5_patch_size", 256),
        transforms=make_transforms(
            train=False,
            pad_to_size=data_cfg.get("pad_to_size"),
            ignore_index=cfg["ignore_index"],
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        drop_last=False,
        pin_memory=False,
    )

    sample_batch = next(iter(val_loader))
    sample_images, _ = _to_numpy_batch(sample_batch)

    state, _, metadata = create_state(
        cfg,
        jax.random.PRNGKey(cfg["seed"]),
        sample_images.shape[1:],
        steps_per_epoch=max(1, len(val_loader)),
    )

    # Checkpoint serialization in train.py stores full config and validates keys strictly.
    # When eval adds extra runtime keys (e.g. data.input_channel_indices), direct restore
    # with runtime cfg may fail due to key mismatch. Reuse checkpoint's own config/metadata
    # as template while restoring model state into current shape-inferred state.
    raw_payload = flax.serialization.msgpack_restore(checkpoint_path.read_bytes())
    ckpt_cfg = raw_payload.get("config", cfg)
    ckpt_metadata = raw_payload.get("metadata", metadata)
    restore_template = {
        "state": state,
        "epoch": 0,
        "best_miou": float("-inf"),
        "metadata": ckpt_metadata,
        "config": ckpt_cfg,
    }
    restored = flax.serialization.from_bytes(restore_template, checkpoint_path.read_bytes())
    state = restored["state"]

    eval_step_jit = jax.jit(
        eval_step,
        static_argnames=(
            "num_classes",
            "ignore_index",
            "primary_loss",
            "lambda_uncertainty",
            "focal_alpha",
            "focal_gamma",
            "ce_weight",
            "dice_weight",
        ),
    )

    conf_mat = np.zeros((cfg["num_classes"], cfg["num_classes"]), dtype=np.int64)
    iterator = tqdm(val_loader, desc="eval", leave=True, dynamic_ncols=True, mininterval=0.5)
    sample_count = 0
    for batch in iterator:
        images_np, labels_np = _to_numpy_batch(batch)
        images = jnp.asarray(images_np)
        preds = eval_step_jit(
            state,
            images,
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
            primary_loss=cfg["loss"].get("primary_loss", "attention_ce"),
            lambda_uncertainty=cfg["loss"].get("lambda_uncertainty", 0.5),
            focal_alpha=cfg["loss"].get("focal_alpha", 0.25),
            focal_gamma=cfg["loss"].get("focal_gamma", 2.0),
            ce_weight=cfg["loss"].get("ce_weight", cfg["loss"].get("focal_weight", 0.5)),
            dice_weight=cfg["loss"].get("dice_weight", 0.5),
        )
        preds_np = np.asarray(jax.device_get(jax.block_until_ready(preds)))
        conf_mat += confusion_matrix(preds_np.reshape(-1), labels_np.reshape(-1), cfg["num_classes"])
        sample_count += int(labels_np.shape[0])

    if sample_count == 0:
        raise RuntimeError("Validation loader is empty")

    metrics = evaluate(conf_mat)
    summary = summarize_metrics(metrics, list(val_ds.class_names))
    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": val_root,
        "num_samples": sample_count,
        "metrics": summary,
        "raw_metrics": _jsonify(metrics),
    }
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2, ensure_ascii=False)
    print(json.dumps(_jsonify(payload), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
