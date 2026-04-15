from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze
from flax.training import train_state
from tqdm import tqdm

from scripts.cloudseg.data import build_dataloaders
from scripts.cloudseg.losses import cloudseg_loss
from scripts.cloudseg.metrics import confusion_matrix, evaluate
from scripts.cloudseg.model import CloudSegAdapter, OfficialUNet, default_backbone_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.json"


class TrainState(train_state.TrainState):
    batch_stats: Any = None


def _to_numpy_batch(batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    images = batch["pixel_values"].numpy().astype(np.float32)
    labels = batch["labels"].numpy().astype(np.int32)
    images = np.transpose(images, (0, 2, 3, 1))
    return images, labels


def _freeze_backbone_grads(grads):
    grads_mut = unfreeze(grads)
    if "LightningDiT_0" in grads_mut:
        grads_mut["LightningDiT_0"] = jax.tree.map(jnp.zeros_like, grads_mut["LightningDiT_0"])
    return freeze(grads_mut)


def _normalize_backbone_cfg(backbone_cfg):
    cfg = dict(backbone_cfg)
    dtype_name = cfg.pop("dtype", "float32")
    if isinstance(dtype_name, str):
        cfg["dtype"] = jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32
    else:
        cfg["dtype"] = dtype_name
    cfg["param_dtype"] = jnp.float32
    return cfg


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "ckpt": output_dir / "ckpt",
        "metric": output_dir / "metric",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2)


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


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in metrics.items():
        flat[f"{prefix}/{key}"] = float(value) if np.isscalar(value) else value
    return flat


def summarize_metrics(metrics: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    per_class_iou = {
        class_names[i]: float(metrics["iou"][i])
        for i in range(min(len(class_names), len(metrics["iou"])))
    }
    per_class_acc = {
        class_names[i]: float(metrics["acc_per_class"][i])
        for i in range(min(len(class_names), len(metrics["acc_per_class"])))
    }
    return {
        "mIoU": float(metrics["mean_iou"]),
        "pixel_acc": float(metrics["acc"]),
        "f1_score": float(metrics["f1_score"]),
        "per_class_iou": per_class_iou,
        "per_class_acc": per_class_acc,
    }


def maybe_init_wandb(cfg: dict[str, Any], output_dir: Path, enabled: bool):
    if not enabled:
        return None
    import wandb

    wandb_cfg = dict(cfg.get("wandb", {}))
    return wandb.init(
        project=wandb_cfg.get("project", "drifting-cloudseg"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name") or output_dir.name,
        dir=str(output_dir),
        config=cfg,
    )


def _checkpoint_payload(state, epoch, best_miou, metadata, cfg):
    return {
        "state": state,
        "epoch": int(epoch),
        "best_miou": float(best_miou),
        "metadata": metadata,
        "config": cfg,
    }


def save_named_checkpoint(path: Path, state, epoch, best_miou, metadata, cfg) -> None:
    payload = _checkpoint_payload(state, epoch, best_miou, metadata, cfg)
    path.write_bytes(flax.serialization.to_bytes(payload))


def restore_named_checkpoint(path: Path, state, metadata, cfg):
    template = _checkpoint_payload(state, 0, -np.inf, metadata, cfg)
    return flax.serialization.from_bytes(template, path.read_bytes())


def create_learning_rate_schedule(cfg, steps_per_epoch: int):
    total_steps = int(cfg["train"]["epochs"]) * int(steps_per_epoch)
    warmup_ratio = float(cfg["optimizer"].get("warmup_ratio", 0.05))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    peak_lr = float(cfg["optimizer"]["learning_rate"])
    min_lr = float(cfg["optimizer"].get("min_learning_rate", 1e-6))
    return optax.warmup_cosine_decay_schedule(
        init_value=min_lr,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=min_lr,
    )


def create_state(cfg, rng, sample_image_shape, steps_per_epoch: int):
    metadata = {}
    model_type = cfg["model"].get("type", "drifting")

    if model_type == "drifting":
        backbone_cfg = _normalize_backbone_cfg(cfg["model"].get("backbone", default_backbone_config()))
        model = CloudSegAdapter(
            backbone_cfg=backbone_cfg,
            num_classes=cfg["num_classes"],
            input_channels=sample_image_shape[-1],
            adapter_hidden_channels=cfg["model"]["adapter_hidden_channels"],
            head_hidden_channels=cfg["model"]["head_hidden_channels"],
            use_input_adapter=cfg["model"].get("use_input_adapter", True),
        )
    elif model_type == "official_unet":
        model = OfficialUNet(
            input_channels=sample_image_shape[-1],
            num_classes=cfg["num_classes"],
        )
    else:
        raise ValueError(f"Unsupported model.type={model_type!r}")

    dummy = jnp.zeros((1, *sample_image_shape), dtype=jnp.float32)
    variables = model.init(rng, dummy, deterministic=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats")
    params = freeze(unfreeze(params))
    learning_rate = create_learning_rate_schedule(cfg, steps_per_epoch)
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer, batch_stats=batch_stats)
    return state, model, metadata


def train_step(
    state,
    images,
    labels,
    *,
    num_classes,
    ignore_index,
    lambda_uncertainty,
    ce_weight,
    dice_weight,
    freeze_backbone,
):
    def loss_fn(params):
        variables = {"params": params}
        if state.batch_stats is not None:
            variables["batch_stats"] = state.batch_stats
            logits, updates = state.apply_fn(variables, images, deterministic=False, mutable=["batch_stats"])
            new_batch_stats = updates["batch_stats"]
        else:
            logits = state.apply_fn(variables, images, deterministic=False)
            new_batch_stats = None
        loss, metrics = cloudseg_loss(
            logits,
            labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
            lambda_uncertainty=lambda_uncertainty,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )
        return loss, (metrics, logits, new_batch_stats)

    (_, (metrics, logits, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = freeze(unfreeze(grads))
    if freeze_backbone:
        grads = _freeze_backbone_grads(grads)
    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)
    preds = jnp.argmax(logits, axis=-1)
    return state, metrics, preds


def eval_step(state, images, *, num_classes, ignore_index, lambda_uncertainty, ce_weight, dice_weight):
    variables = {"params": state.params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats
    logits = state.apply_fn(variables, images, deterministic=True)
    preds = jnp.argmax(logits, axis=-1)
    return preds


def run_train_epoch(loader, state, step_fn, *, split, cfg):
    conf_mat = np.zeros((cfg["num_classes"], cfg["num_classes"]), dtype=np.int64)
    metric_sums = {"loss": 0.0, "loss_ce": 0.0, "loss_dice": 0.0}
    count = 0

    iterator = tqdm(loader, desc=split, leave=True, dynamic_ncols=True, mininterval=0.5)
    for batch in iterator:
        images_np, labels_np = _to_numpy_batch(batch)
        images = jnp.asarray(images_np)
        labels = jnp.asarray(labels_np)
        state, metrics, preds = step_fn(
            state,
            images,
            labels,
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
            lambda_uncertainty=cfg["loss"]["lambda_uncertainty"],
            ce_weight=cfg["loss"]["ce_weight"],
            dice_weight=cfg["loss"]["dice_weight"],
            freeze_backbone=not cfg["train_backbone"],
        )
        preds_np = np.asarray(preds)
        labels_eval = np.asarray(labels)
        conf_mat += confusion_matrix(preds_np.reshape(-1), labels_eval.reshape(-1), cfg["num_classes"])
        batch_size = labels_np.shape[0]
        count += batch_size
        for key in metric_sums:
            metric_sums[key] += float(metrics[key]) * batch_size
        iterator.set_postfix(loss=f"{float(metrics['loss']):.4f}")

    epoch_metrics = evaluate(conf_mat)
    for key in metric_sums:
        epoch_metrics[key] = metric_sums[key] / max(count, 1)
    return state, epoch_metrics


def run_eval_epoch(loader, state, step_fn, *, split, cfg):
    conf_mat = np.zeros((cfg["num_classes"], cfg["num_classes"]), dtype=np.int64)
    iterator = tqdm(loader, desc=split, leave=True, dynamic_ncols=True, mininterval=0.5)
    for batch in iterator:
        images_np, labels_np = _to_numpy_batch(batch)
        images = jnp.asarray(images_np)
        preds = step_fn(
            state,
            images,
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
            lambda_uncertainty=cfg["loss"]["lambda_uncertainty"],
            ce_weight=cfg["loss"]["ce_weight"],
            dice_weight=cfg["loss"]["dice_weight"],
        )
        preds_np = np.asarray(jax.device_get(jax.block_until_ready(preds)))
        conf_mat += confusion_matrix(preds_np.reshape(-1), labels_np.reshape(-1), cfg["num_classes"])
        del images, preds, preds_np
    return evaluate(conf_mat)


def build_parser():
    parser = argparse.ArgumentParser(description="Cloud segmentation adapter using drifting backbone.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.resume:
        cfg["resume"] = args.resume
    cfg["wandb_enabled"] = bool(args.wandb)

    output_dir = Path(cfg["output_dir"]).resolve()
    output_paths = setup_output_dirs(output_dir)
    write_json(output_paths["root"] / "resolved_config.json", cfg)

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
    )

    print("[cloudseg] preparing sample batch for shape inference...", flush=True)
    sample_batch = next(iter(train_loader))
    sample_images, _ = _to_numpy_batch(sample_batch)

    rng = jax.random.PRNGKey(cfg["seed"])
    print("[cloudseg] initializing model from config...", flush=True)
    state, _, metadata = create_state(cfg, rng, sample_images.shape[1:], steps_per_epoch=len(train_loader))
    start_epoch = 0
    best_miou = -np.inf

    resume_path = cfg.get("resume", "")
    if resume_path:
        print(f"[cloudseg] resuming training state from {resume_path}...", flush=True)
        restored = restore_named_checkpoint(Path(resume_path), state, metadata, cfg)
        state = restored["state"]
        start_epoch = int(restored["epoch"]) + 1
        best_miou = float(restored["best_miou"])
        metadata = restored["metadata"]

    train_step_jit = jax.jit(
        train_step,
        static_argnames=("num_classes", "ignore_index", "lambda_uncertainty", "ce_weight", "dice_weight", "freeze_backbone"),
    )
    eval_step_jit = jax.jit(
        eval_step,
        static_argnames=("num_classes", "ignore_index", "lambda_uncertainty", "ce_weight", "dice_weight"),
    )

    wandb_run = maybe_init_wandb(cfg, output_dir, cfg["wandb_enabled"])
    save_every = int(cfg["train"]["save_every_epochs"])
    eval_every = int(cfg["eval"]["eval_every_epochs"])
    total_epochs = int(cfg["train"]["epochs"])

    print("[cloudseg] compiling first train step on first batch; this can take a while...", flush=True)
    history = []
    first_train_compile_timed = False
    for epoch in range(start_epoch, total_epochs):
        print(f"[cloudseg] epoch {epoch + 1}/{total_epochs}", flush=True)
        compile_start = None
        if not first_train_compile_timed:
            compile_start = time.perf_counter()
        state, train_metrics = run_train_epoch(train_loader, state, train_step_jit, split=f"train[{epoch}]", cfg=cfg)
        if not first_train_compile_timed and compile_start is not None:
            compile_elapsed = time.perf_counter() - compile_start
            print(f"[cloudseg] first train epoch entered after JAX compile in {compile_elapsed:.2f}s", flush=True)
            first_train_compile_timed = True
        train_summary = summarize_metrics(train_metrics, train_ds.class_names)
        record = {
            "epoch": epoch,
            "train": {
                "loss": float(train_metrics["loss"]),
                "lr": float(create_learning_rate_schedule(cfg, len(train_loader))(state.step)),
            },
        }

        if (epoch + 1) % eval_every == 0:
            val_metrics = run_eval_epoch(val_loader, state, eval_step_jit, split=f"val[{epoch}]", cfg=cfg)
            val_summary = summarize_metrics(val_metrics, train_ds.class_names)
            record["val"] = val_summary
            metrics_out = {
                "epoch": epoch,
                "val": val_summary,
            }
            write_json(output_paths["metric"] / f"epoch_{epoch:04d}.json", metrics_out)
            if val_metrics["mean_iou"] > best_miou:
                best_miou = float(val_metrics["mean_iou"])
                save_named_checkpoint(output_paths["ckpt"] / "best.ckpt", state, epoch, best_miou, metadata, cfg)
                write_json(
                    output_paths["metric"] / "best.json",
                    {
                        "best_epoch": epoch,
                        "criterion": "mIoU",
                        "best_mIoU": float(val_summary["mIoU"]),
                        "val": val_summary,
                    },
                )

        history.append(record)
        if "val" in record:
            print(
                f"[cloudseg] epoch {epoch + 1} val mIoU={record['val']['mIoU']:.4f} "
                f"pixel_acc={record['val']['pixel_acc']:.4f}",
                flush=True,
            )
        else:
            print(
                f"[cloudseg] epoch {epoch + 1} train loss={record['train']['loss']:.4f} "
                f"lr={record['train']['lr']:.6e}",
                flush=True,
            )

        save_named_checkpoint(output_paths["ckpt"] / "latest.ckpt", state, epoch, best_miou, metadata, cfg)
        if (epoch + 1) % save_every == 0:
            save_named_checkpoint(output_paths["ckpt"] / f"epoch_{epoch:04d}.ckpt", state, epoch, best_miou, metadata, cfg)

        if wandb_run is not None:
            payload = {"epoch": epoch}
            payload["train/loss"] = float(train_metrics["loss"])
            payload["train/lr"] = record["train"]["lr"]
            if "val" in record:
                payload.update(flatten_metrics("val", record["val"]))
            wandb_run.log(payload, step=epoch)

    write_json(output_paths["root"] / "history.json", {"history": history})
    write_json(output_paths["root"] / "metadata.json", metadata)
    write_json(output_paths["root"] / "class_names.json", {"class_names": train_ds.class_names})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
