from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from scripts.cloudseg.data import build_dataloaders
from scripts.cloudseg.metrics import confusion_matrix, evaluate
from scripts.cloudseg.train import create_state, eval_step, load_config, restore_named_checkpoint, summarize_metrics


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "unet_official.json"

CLOUD_CLASS_COLORS = np.array(
    [
        [0xFF, 0xFF, 0xFF],
        [0xCC, 0xCC, 0xFF],
        [0x66, 0x66, 0xFF],
        [0x00, 0x00, 0xFF],
        [0xFF, 0xFF, 0xCC],
        [0xFF, 0xFF, 0x00],
        [0xCC, 0xCC, 0x00],
        [0xFF, 0x99, 0x99],
        [0xFF, 0x66, 0x33],
        [0xFF, 0x00, 0x00],
    ],
    dtype=np.uint8,
)


def _to_numpy_batch(batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    images = batch["pixel_values"].numpy().astype(np.float32)
    labels = batch["labels"].numpy().astype(np.int32)
    images = np.transpose(images, (0, 2, 3, 1))
    return images, labels


def _safe_filename(value: str, fallback: str) -> str:
    stem = os.path.splitext(os.path.basename(str(value)))[0] or fallback
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem[:180] or fallback


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


def _colorize_mask(mask: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    colors = CLOUD_CLASS_COLORS
    if num_classes > len(colors):
        rng = np.random.default_rng(0)
        extra = rng.integers(0, 256, size=(num_classes - len(colors), 3), dtype=np.uint8)
        colors = np.concatenate([colors, extra], axis=0)

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < num_classes)
    rgb[valid] = colors[mask[valid]]
    rgb[mask == ignore_index] = np.array([0, 0, 0], dtype=np.uint8)
    return rgb


def _write_sample_visualization(
    path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    class_names: list[str],
    iou_per_class: np.ndarray,
    acc_per_class: np.ndarray,
    reference_iou_per_class: np.ndarray,
    reference_acc_per_class: np.ndarray,
    sample_title: str,
    ignore_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = len(class_names)
    gt_rgb = _colorize_mask(gt, num_classes, ignore_index)
    pred_rgb = _colorize_mask(pred, num_classes, ignore_index)

    fig = plt.figure(figsize=(9.8, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.18, 0.95])
    ax_gt = fig.add_subplot(grid[0, 0])
    ax_pred = fig.add_subplot(grid[0, 1])
    ax_colorbar = fig.add_subplot(grid[1, :])
    ax_bar = fig.add_subplot(grid[2, :])

    ax_gt.imshow(gt_rgb)
    ax_gt.set_title("GT")
    ax_gt.axis("off")
    ax_pred.imshow(pred_rgb)
    ax_pred.set_title("Pred")
    ax_pred.axis("off")

    color_strip = np.arange(num_classes, dtype=np.int64)[None, :]
    ax_colorbar.imshow(_colorize_mask(color_strip, num_classes, ignore_index), aspect="auto")
    ax_colorbar.set_xticks(np.arange(num_classes))
    ax_colorbar.set_xticklabels(class_names, fontsize=9)
    ax_colorbar.set_yticks([])
    ax_colorbar.set_title("Class Color Map", fontsize=10)
    for spine in ax_colorbar.spines.values():
        spine.set_visible(False)

    y = np.arange(num_classes)
    ax_bar.barh(y - 0.18, iou_per_class, height=0.34, color="#3B82F6", label="mIoU")
    ax_bar.barh(y + 0.18, acc_per_class, height=0.34, color="#F97316", label="Pixel Acc")
    ax_bar.scatter(
        reference_iou_per_class,
        y - 0.18,
        marker="|",
        s=120,
        color="#111827",
        linewidths=1.8,
        label="Eval Avg",
        zorder=4,
    )
    ax_bar.scatter(
        reference_acc_per_class,
        y + 0.18,
        marker="|",
        s=120,
        color="#111827",
        linewidths=1.8,
        zorder=4,
    )
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(class_names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, 1.18)
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.legend(loc="lower right", fontsize=9)
    ax_bar.set_title("10 Cloud Classes Metrics vs Eval Average")
    ax_bar.set_xlabel("score")

    for idx in range(num_classes):
        delta_iou = float(iou_per_class[idx] - reference_iou_per_class[idx])
        delta_acc = float(acc_per_class[idx] - reference_acc_per_class[idx])
        better = (iou_per_class[idx] >= reference_iou_per_class[idx]) and (
            acc_per_class[idx] >= reference_acc_per_class[idx]
        )
        color = "#15803D" if better else "#B91C1C"
        ax_bar.text(
            1.02,
            idx,
            f"mIoU {'+' if delta_iou >= 0 else ''}{delta_iou:.2f} / Acc {'+' if delta_acc >= 0 else ''}{delta_acc:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=color,
        )
    fig.suptitle(sample_title, fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_records(records: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    jsonl_path = output_dir / "per_sample_metrics.jsonl"
    csv_path = output_dir / "per_sample_metrics.csv"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    class_names = records[0]["class_names"] if records else []
    fieldnames = ["index", "sample_id", "source_path", "pixel_acc", "mean_acc", "mean_iou", "png_path"]
    for class_name in class_names:
        fieldnames.extend([f"{class_name}_miou", f"{class_name}_pixel_acc"])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "index": record["index"],
                "sample_id": record["sample_id"],
                "source_path": record["source_path"],
                "pixel_acc": record["pixel_acc"],
                "mean_acc": record["mean_acc"],
                "mean_iou": record["mean_iou"],
                "png_path": record["png_path"],
            }
            for class_name, class_iou, class_acc in zip(
                record["class_names"],
                record["iou_per_class"],
                record["acc_per_class"],
            ):
                row[f"{class_name}_miou"] = class_iou
                row[f"{class_name}_pixel_acc"] = class_acc
            writer.writerow(row)
    return {"jsonl": str(jsonl_path), "csv": str(csv_path)}


def _save_confusion_matrix_artifacts(conf_mat: np.ndarray, class_names: list[str], output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_counts_path = output_dir / "confusion_matrix_counts.csv"
    cm_norm_path = output_dir / "confusion_matrix_row_normalized.csv"
    cm_png_path = output_dir / "confusion_matrix.png"

    conf_mat = conf_mat.astype(np.int64, copy=False)
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    conf_norm = np.divide(
        conf_mat.astype(np.float64),
        np.maximum(row_sums, 1),
        where=row_sums > 0,
    )

    with open(cm_counts_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *class_names])
        for idx, name in enumerate(class_names):
            writer.writerow([name, *conf_mat[idx].tolist()])

    with open(cm_norm_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *class_names])
        for idx, name in enumerate(class_names):
            writer.writerow([name, *[f"{v:.6f}" for v in conf_norm[idx]]])

    fig, ax = plt.subplots(figsize=(10.2, 8.4), constrained_layout=True)
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized ratio", rotation=90)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Confusion Matrix (Row-normalized)")

    for i in range(conf_norm.shape[0]):
        for j in range(conf_norm.shape[1]):
            text_color = "white" if conf_norm[i, j] >= 0.5 else "black"
            ax.text(j, i, f"{conf_norm[i, j]:.2f}\n({int(conf_mat[i, j])})", ha="center", va="center", fontsize=7, color=text_color)

    fig.savefig(cm_png_path, dpi=180)
    plt.close(fig)
    return {
        "confusion_matrix_png": str(cm_png_path),
        "confusion_matrix_counts_csv": str(cm_counts_path),
        "confusion_matrix_row_normalized_csv": str(cm_norm_path),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate best UNet checkpoint and save per-sample visualizations.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--vis-output-dir", type=str, default="")
    parser.add_argument("--vis-max-samples", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else (output_dir / "ckpt" / "best.ckpt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    vis_output_dir = Path(args.vis_output_dir).resolve() if args.vis_output_dir else (output_dir / "visualizations")
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    print("[cloudseg] building dataloaders...", flush=True)
    train_loader, val_loader, train_ds, _ = build_dataloaders(
        cfg["data"]["root"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        batch_size=cfg["train"]["batch_size"],
        eval_batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        ignore_index=cfg["ignore_index"],
        pad_to_size=cfg["data"].get("pad_to_size"),
        normalization=cfg["data"].get("normalization"),
        input_channel_indices=cfg["data"].get("input_channel_indices"),
        feature_augmentation=cfg["data"].get("feature_augmentation"),
        h5_patch_size=cfg["data"].get("h5_patch_size", 256),
    )

    sample_batch = next(iter(train_loader))
    sample_images, _ = _to_numpy_batch(sample_batch)
    state, _, metadata = create_state(
        cfg,
        jax.random.PRNGKey(cfg["seed"]),
        sample_images.shape[1:],
        steps_per_epoch=max(1, len(train_loader)),
    )
    restored = restore_named_checkpoint(checkpoint_path, state, metadata, cfg)
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

    num_classes = int(cfg["num_classes"])
    ignore_index = int(cfg["ignore_index"])
    class_names = list(train_ds.class_names)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    records: list[dict[str, Any]] = []
    pending_visualizations: list[dict[str, Any]] = []
    sample_index = 0

    iterator = tqdm(val_loader, desc="eval+vis", leave=True, dynamic_ncols=True, mininterval=0.5)
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
        sample_ids = list(batch.get("sample_id", [str(i) for i in range(sample_index, sample_index + len(preds_np))]))
        source_paths = list(batch.get("source_path", [""] * len(preds_np)))

        for batch_idx, (pred, label) in enumerate(zip(preds_np, labels_np)):
            sample_conf = confusion_matrix(pred.reshape(-1), label.reshape(-1), num_classes)
            conf_mat += sample_conf
            sample_metrics = evaluate(sample_conf)
            sample_id = str(sample_ids[batch_idx])
            source_path = str(source_paths[batch_idx])

            png_path = ""
            if args.vis_max_samples is None or sample_index < args.vis_max_samples:
                filename = f"{sample_index:06d}_{_safe_filename(sample_id, str(sample_index))}.png"
                sample_png_path = vis_output_dir / filename
                png_path = str(sample_png_path)
                pending_visualizations.append(
                    {
                        "path": sample_png_path,
                        "gt": label.copy(),
                        "pred": pred.copy(),
                        "iou_per_class": sample_metrics["iou"].copy(),
                        "acc_per_class": sample_metrics["acc_per_class"].copy(),
                        "sample_title": sample_id,
                    }
                )

            records.append(
                {
                    "index": sample_index,
                    "sample_id": sample_id,
                    "source_path": source_path,
                    "pixel_acc": float(sample_metrics["acc"]),
                    "mean_acc": float(sample_metrics["pre"]),
                    "mean_iou": float(sample_metrics["mean_iou"]),
                    "class_names": class_names,
                    "iou_per_class": sample_metrics["iou"].astype(float).tolist(),
                    "acc_per_class": sample_metrics["acc_per_class"].astype(float).tolist(),
                    "png_path": png_path,
                }
            )
            sample_index += 1
        del images, preds, preds_np

    if sample_index == 0:
        raise RuntimeError("Validation loader is empty")

    metrics = evaluate(conf_mat)
    for visual in tqdm(
        pending_visualizations,
        desc="save vis",
        leave=False,
        dynamic_ncols=True,
        disable=not pending_visualizations,
    ):
        _write_sample_visualization(
            visual["path"],
            gt=visual["gt"],
            pred=visual["pred"],
            class_names=class_names,
            iou_per_class=visual["iou_per_class"],
            acc_per_class=visual["acc_per_class"],
            reference_iou_per_class=metrics["iou"],
            reference_acc_per_class=metrics["acc_per_class"],
            sample_title=visual["sample_title"],
            ignore_index=ignore_index,
        )

    record_paths = _save_records(records, output_dir)
    cm_paths = _save_confusion_matrix_artifacts(conf_mat, class_names, output_dir)
    summary = summarize_metrics(metrics, class_names)
    payload = {
        "checkpoint": str(checkpoint_path),
        "metrics": summary,
        "raw_metrics": _jsonify(metrics),
        "visualization_dir": str(vis_output_dir),
        "per_sample_metrics_jsonl": record_paths["jsonl"],
        "per_sample_metrics_csv": record_paths["csv"],
        "visualized_samples": min(sample_index, args.vis_max_samples) if args.vis_max_samples is not None else sample_index,
        **cm_paths,
    }
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2, ensure_ascii=False)
    print(json.dumps(_jsonify(payload), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
