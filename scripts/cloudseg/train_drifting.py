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

from drift_loss import drift_loss
from memory_bank import ArrayMemoryBank
from scripts.cloudseg.data import build_dataloaders
from scripts.cloudseg.losses import cloudseg_loss
from scripts.cloudseg.metrics import confusion_matrix, evaluate
from scripts.cloudseg.model import DriftingUNet


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "unet_drifting.json"


class TrainState(train_state.TrainState):
    batch_stats: Any = None


def _to_numpy_batch(batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    images = batch["pixel_values"].numpy().astype(np.float32)
    labels = batch["labels"].numpy().astype(np.int32)
    images = np.transpose(images, (0, 2, 3, 1))
    return images, labels


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


def summarize_metrics(metrics: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    per_class_iou = {
        class_names[i]: float(metrics["iou"][i])
        for i in range(min(len(class_names), len(metrics["iou"])))
    }
    per_class_acc = {
        class_names[i]: float(metrics["acc_per_class"][i])
        for i in range(min(len(class_names), len(metrics["acc_per_class"])))
    }
    summary = {
        "mIoU": float(metrics["mean_iou"]),
        "pixel_acc": float(metrics["acc"]),
        "f1_score": float(metrics["f1_score"]),
        "per_class_iou": per_class_iou,
        "per_class_acc": per_class_acc,
    }
    for key, value in metrics.items():
        if key in {"iou", "acc_per_class", "mean_iou", "acc", "f1_score", "pre", "kappa", "recall"}:
            continue
        if is_scalar_metric(value):
            summary[key] = float(value)
    return summary


def split_summary_metrics(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    main = {}
    aux = {}
    for key, value in summary.items():
        if key in {"mIoU", "pixel_acc", "f1_score", "per_class_iou", "per_class_acc"}:
            main[key] = value
        else:
            aux[key] = value
    return main, aux


def metric_value(container: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = container.get(key, default)
    return float(value) if np.isscalar(value) else float(default)


def is_scalar_metric(value: Any) -> bool:
    try:
        return np.asarray(value).ndim == 0
    except Exception:
        return False


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update(flatten_metrics(f"{prefix}/{key}", value))
        else:
            flat[f"{prefix}/{key}"] = float(value) if is_scalar_metric(value) else value
    return flat


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


def serialize_bank(bank: ArrayMemoryBank) -> dict[str, Any]:
    return {
        "num_classes": int(bank.num_classes),
        "max_size": int(bank.max_size),
        "dtype": str(np.dtype(bank.dtype).name),
        "feature_shape": tuple(bank.feature_shape) if bank.feature_shape is not None else None,
        "ptr": np.asarray(bank.ptr, dtype=np.int32),
        "count": np.asarray(bank.count, dtype=np.int32),
        "bank": None if bank.bank is None else np.asarray(bank.bank),
    }


def deserialize_bank(payload: dict[str, Any]) -> ArrayMemoryBank:
    bank = ArrayMemoryBank(
        num_classes=int(payload["num_classes"]),
        max_size=int(payload["max_size"]),
        dtype=np.dtype(payload["dtype"]),
    )
    feature_shape = payload.get("feature_shape")
    if feature_shape is not None:
        bank.feature_shape = tuple(int(v) for v in feature_shape)
    bank.ptr = np.asarray(payload["ptr"], dtype=np.int32)
    bank.count = np.asarray(payload["count"], dtype=np.int32)
    stored = payload.get("bank")
    if stored is not None:
        bank.bank = np.asarray(stored, dtype=bank.dtype)
    return bank


def serialize_banks(banks: dict[str, ArrayMemoryBank]) -> dict[str, Any]:
    return {level: serialize_bank(bank) for level, bank in banks.items()}


def deserialize_banks(payload: dict[str, Any]) -> dict[str, ArrayMemoryBank]:
    return {level: deserialize_bank(bank_payload) for level, bank_payload in payload.items()}


def resolve_drift_class_ids(class_spec, class_names: list[str], *, num_classes: int) -> list[int]:
    if not class_spec:
        return list(range(num_classes))

    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    resolved = []
    for item in class_spec:
        if isinstance(item, str):
            if item not in name_to_idx:
                raise ValueError(
                    f"Unknown drift class name {item!r}. Available names: {', '.join(class_names)}"
                )
            resolved.append(name_to_idx[item])
        else:
            idx = int(item)
            if idx < 0 or idx >= num_classes:
                raise ValueError(f"Drift class id {idx} is out of range [0, {num_classes - 1}]")
            resolved.append(idx)
    return sorted(set(resolved))


def _checkpoint_payload(state, epoch, best_miou, metadata, cfg, positive_banks, negative_banks) -> dict[str, Any]:
    return {
        "state": state,
        "epoch": int(epoch),
        "best_miou": float(best_miou),
        "metadata": metadata,
        "config": cfg,
        "positive_banks": serialize_banks(positive_banks),
        "negative_banks": serialize_banks(negative_banks),
    }


def save_named_checkpoint(path: Path, state, epoch, best_miou, metadata, cfg, positive_banks, negative_banks) -> None:
    payload = _checkpoint_payload(state, epoch, best_miou, metadata, cfg, positive_banks, negative_banks)
    path.write_bytes(flax.serialization.to_bytes(payload))


def restore_named_checkpoint(path: Path, state, metadata, cfg, positive_banks, negative_banks):
    template = _checkpoint_payload(state, 0, -np.inf, metadata, cfg, positive_banks, negative_banks)
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
    model = DriftingUNet(
        input_channels=sample_image_shape[-1],
        num_classes=cfg["num_classes"],
    )
    dummy = jnp.zeros((1, *sample_image_shape), dtype=jnp.float32)
    variables = model.init(rng, dummy, deterministic=True, return_features=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats")
    learning_rate = create_learning_rate_schedule(cfg, steps_per_epoch)
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    state = TrainState.create(apply_fn=model.apply, params=freeze(unfreeze(params)), tx=optimizer, batch_stats=batch_stats)
    return state, model, metadata


def _resize_labels(labels: jax.Array, height: int, width: int) -> jax.Array:
    resized = jax.image.resize(
        labels[..., None].astype(jnp.float32),
        (labels.shape[0], height, width, 1),
        method="nearest",
    )
    return resized[..., 0].astype(jnp.int32)


def extract_class_prototypes(features: dict[str, jax.Array], labels: jax.Array, *, num_classes: int, ignore_index: int):
    prototypes = {}
    valids = {}
    for level, feat in features.items():
        level_labels = _resize_labels(labels, feat.shape[1], feat.shape[2])
        valid_mask = level_labels != ignore_index
        safe_labels = jnp.clip(level_labels, 0, num_classes - 1)
        one_hot = jax.nn.one_hot(safe_labels, num_classes, dtype=feat.dtype) * valid_mask[..., None].astype(feat.dtype)
        class_sum = jnp.einsum("bhwc,bhwk->bkc", feat, one_hot)
        class_count = jnp.sum(one_hot, axis=(1, 2))
        denom = jnp.maximum(class_count[..., None], 1.0)
        proto = class_sum / denom
        valid = class_count > 0
        proto = proto * valid[..., None].astype(proto.dtype)
        prototypes[level] = proto
        valids[level] = valid
    return prototypes, valids


def _build_bank_bundle(level_dims: dict[str, int], *, num_classes: int, positive_bank_size: int, negative_bank_size: int):
    positive = {}
    negative = {}
    for level, dim in level_dims.items():
        positive[level] = ArrayMemoryBank(num_classes=num_classes, max_size=positive_bank_size, dtype=np.float32)
        negative[level] = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size, dtype=np.float32)
        positive[level]._init_bank((dim,))
        negative[level]._init_bank((dim,))
    return positive, negative


def sample_bank_targets(
    positive_banks,
    negative_banks,
    *,
    batch_size: int,
    num_classes: int,
    pos_per_class: int,
    neg_per_class: int,
    drift_class_ids: np.ndarray,
):
    class_ids = np.tile(drift_class_ids.astype(np.int32), batch_size)
    sampled_pos = {}
    sampled_neg = {}
    for level in positive_banks:
        sampled_pos[level] = positive_banks[level].sample(class_ids, pos_per_class).astype(jnp.float32)
        sampled_neg[level] = negative_banks[level].sample(np.zeros_like(class_ids), neg_per_class).astype(jnp.float32)
    return sampled_pos, sampled_neg


def update_memory_banks(positive_banks, negative_banks, prototypes, valids) -> None:
    for level, proto in prototypes.items():
        proto_np = np.asarray(proto, dtype=np.float32)
        valid_np = np.asarray(valids[level], dtype=bool)
        batch_size, num_classes, dim = proto_np.shape
        flat_proto = proto_np.reshape(batch_size * num_classes, dim)
        flat_valid = valid_np.reshape(batch_size * num_classes)
        class_ids = np.tile(np.arange(num_classes, dtype=np.int32), batch_size)
        if not np.any(flat_valid):
            continue
        valid_proto = flat_proto[flat_valid]
        valid_class_ids = class_ids[flat_valid]
        positive_banks[level].add(valid_proto, valid_class_ids)
        negative_banks[level].add(valid_proto, np.zeros(valid_proto.shape[0], dtype=np.int32))


def drift_ready(positive_banks, negative_banks, *, pos_per_class: int, neg_per_class: int, drift_class_ids: np.ndarray) -> bool:
    for level in positive_banks:
        if np.min(positive_banks[level].count[drift_class_ids]) <= 0:
            return False
        if negative_banks[level].count[0] < max(1, neg_per_class):
            return False
    return True


def _mean_if_any(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(jnp.float32)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def train_step(
    state,
    images,
    labels,
    bank_pos,
    bank_neg,
    *,
    num_classes,
    ignore_index,
    lambda_uncertainty,
    ce_weight,
    dice_weight,
    drift_weight,
    bank_ready,
    drift_r_list,
    drift_class_ids,
):
    def loss_fn(params):
        variables = {"params": params}
        if state.batch_stats is not None:
            variables["batch_stats"] = state.batch_stats
            outputs, updates = state.apply_fn(
                variables,
                images,
                deterministic=False,
                return_features=True,
                mutable=["batch_stats"],
            )
            new_batch_stats = updates["batch_stats"]
        else:
            outputs = state.apply_fn(variables, images, deterministic=False, return_features=True)
            new_batch_stats = None

        logits = outputs["logits"]
        features = outputs["features"]
        seg_loss, seg_metrics = cloudseg_loss(
            logits,
            labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
            lambda_uncertainty=lambda_uncertainty,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )

        prototypes, valids = extract_class_prototypes(
            features,
            labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        allowed_classes = jnp.asarray(drift_class_ids, dtype=jnp.int32)
        allowed_mask = jnp.isin(jnp.arange(num_classes, dtype=jnp.int32), allowed_classes)
        drift_terms = []
        drift_metrics = {}
        for level, proto in prototypes.items():
            flat_proto = proto.reshape(-1, 1, proto.shape[-1])
            level_valid = valids[level] & allowed_mask[None, :]
            flat_valid = level_valid.reshape(-1)

            if bank_ready:
                loss_vec, info = drift_loss(
                    gen=flat_proto,
                    fixed_pos=bank_pos[level],
                    fixed_neg=bank_neg[level],
                    R_list=drift_r_list,
                )
                masked_loss = _mean_if_any(loss_vec, flat_valid)
                drift_terms.append(masked_loss)
                drift_metrics[f"drift_{level}"] = masked_loss
                drift_metrics[f"active_{level}"] = flat_valid.astype(jnp.float32).mean()
                for key, value in info.items():
                    drift_metrics[f"{level}_{key}"] = value
            else:
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                drift_terms.append(zero)
                drift_metrics[f"drift_{level}"] = zero
                drift_metrics[f"active_{level}"] = flat_valid.astype(jnp.float32).mean()

        drift_total = sum(drift_terms) / max(len(drift_terms), 1)
        total = seg_loss + drift_weight * drift_total
        metrics = {
            **seg_metrics,
            "loss": total,
            "loss_seg": seg_loss,
            "loss_drift": drift_total,
            **drift_metrics,
        }
        return total, (metrics, logits, new_batch_stats, prototypes, valids)

    (_, (metrics, logits, new_batch_stats, prototypes, valids)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)
    preds = jnp.argmax(logits, axis=-1)
    return state, metrics, preds, prototypes, valids


def eval_step(state, images):
    variables = {"params": state.params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats
    logits = state.apply_fn(variables, images, deterministic=True, return_features=False)
    preds = jnp.argmax(logits, axis=-1)
    return preds


def infer_level_dims(state, sample_images):
    variables = {"params": state.params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats
    outputs = state.apply_fn(variables, sample_images, deterministic=True, return_features=True)
    features = outputs["features"]
    return {level: int(value.shape[-1]) for level, value in features.items()}


def run_train_epoch(loader, state, step_fn, *, split, cfg, positive_banks, negative_banks):
    conf_mat = np.zeros((cfg["num_classes"], cfg["num_classes"]), dtype=np.int64)
    metric_sums = {}
    count = 0
    drift_cfg = cfg["drift"]
    drift_class_ids = np.asarray(drift_cfg["class_ids"], dtype=np.int32)

    iterator = tqdm(loader, desc=split, leave=True, dynamic_ncols=True, mininterval=0.5)
    for batch in iterator:
        images_np, labels_np = _to_numpy_batch(batch)
        images = jnp.asarray(images_np)
        labels = jnp.asarray(labels_np)
        ready = drift_ready(
            positive_banks,
            negative_banks,
            pos_per_class=drift_cfg["pos_per_class"],
            neg_per_class=drift_cfg["neg_per_class"],
            drift_class_ids=drift_class_ids,
        )
        if ready:
            bank_pos, bank_neg = sample_bank_targets(
                positive_banks,
                negative_banks,
                batch_size=labels_np.shape[0],
                num_classes=cfg["num_classes"],
                pos_per_class=drift_cfg["pos_per_class"],
                neg_per_class=drift_cfg["neg_per_class"],
                drift_class_ids=drift_class_ids,
            )
        else:
            bank_pos = {
                level: jnp.zeros((labels_np.shape[0] * len(drift_class_ids), drift_cfg["pos_per_class"], bank.feature_shape[0]), dtype=jnp.float32)
                for level, bank in positive_banks.items()
            }
            bank_neg = {
                level: jnp.zeros((labels_np.shape[0] * len(drift_class_ids), drift_cfg["neg_per_class"], bank.feature_shape[0]), dtype=jnp.float32)
                for level, bank in negative_banks.items()
            }

        state, metrics, preds, prototypes, valids = step_fn(
            state,
            images,
            labels,
            bank_pos,
            bank_neg,
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
            lambda_uncertainty=cfg["loss"]["lambda_uncertainty"],
            ce_weight=cfg["loss"]["ce_weight"],
            dice_weight=cfg["loss"]["dice_weight"],
            drift_weight=drift_cfg["weight"],
            bank_ready=ready,
            drift_r_list=tuple(drift_cfg["loss_kwargs"]["R_list"]),
            drift_class_ids=tuple(int(v) for v in drift_class_ids.tolist()),
        )
        update_memory_banks(positive_banks, negative_banks, jax.device_get(prototypes), jax.device_get(valids))

        preds_np = np.asarray(jax.device_get(preds))
        labels_eval = np.asarray(labels)
        conf_mat += confusion_matrix(preds_np.reshape(-1), labels_eval.reshape(-1), cfg["num_classes"])
        batch_size = labels_np.shape[0]
        count += batch_size
        for key, value in metrics.items():
            if not is_scalar_metric(value):
                continue
            metric_sums.setdefault(key, 0.0)
            metric_sums[key] += float(value) * batch_size
        iterator.set_postfix(loss=f"{float(metrics['loss']):.4f}", drift=f"{float(metrics['loss_drift']):.4f}")

    epoch_metrics = evaluate(conf_mat)
    for key, value in metric_sums.items():
        epoch_metrics[key] = value / max(count, 1)
    return state, epoch_metrics


def run_eval_epoch(loader, state, step_fn, *, split, cfg):
    conf_mat = np.zeros((cfg["num_classes"], cfg["num_classes"]), dtype=np.int64)
    iterator = tqdm(loader, desc=split, leave=True, dynamic_ncols=True, mininterval=0.5)
    for batch in iterator:
        images_np, labels_np = _to_numpy_batch(batch)
        images = jnp.asarray(images_np)
        preds = step_fn(state, images)
        preds_np = np.asarray(jax.device_get(jax.block_until_ready(preds)))
        conf_mat += confusion_matrix(preds_np.reshape(-1), labels_np.reshape(-1), cfg["num_classes"])
    return evaluate(conf_mat)


def build_parser():
    parser = argparse.ArgumentParser(description="Cloud segmentation with U-Net backbone and drifting supervision.")
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

    print("[cloudseg-drift] building dataloaders...", flush=True)
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
    drift_class_ids = resolve_drift_class_ids(
        cfg["drift"].get("class_ids"),
        train_ds.class_names,
        num_classes=cfg["num_classes"],
    )
    cfg["drift"]["class_ids"] = drift_class_ids
    cfg["drift"]["class_names"] = [train_ds.class_names[idx] for idx in drift_class_ids]
    write_json(output_paths["root"] / "resolved_config.json", cfg)

    sample_batch = next(iter(train_loader))
    sample_images, _ = _to_numpy_batch(sample_batch)
    sample_images_jax = jnp.asarray(sample_images[:1])

    rng = jax.random.PRNGKey(cfg["seed"])
    print("[cloudseg-drift] initializing model...", flush=True)
    state, _, metadata = create_state(cfg, rng, sample_images.shape[1:], steps_per_epoch=len(train_loader))
    level_dims = infer_level_dims(state, sample_images_jax)
    positive_banks, negative_banks = _build_bank_bundle(
        level_dims,
        num_classes=cfg["num_classes"],
        positive_bank_size=cfg["drift"]["positive_bank_size"],
        negative_bank_size=cfg["drift"]["negative_bank_size"],
    )

    start_epoch = 0
    best_miou = -np.inf
    resume_path = cfg.get("resume", "")
    if resume_path:
        print(f"[cloudseg-drift] resuming training state from {resume_path}...", flush=True)
        restored = restore_named_checkpoint(Path(resume_path), state, metadata, cfg, positive_banks, negative_banks)
        state = restored["state"]
        start_epoch = int(restored["epoch"]) + 1
        best_miou = float(restored["best_miou"])
        metadata = restored["metadata"]
        positive_banks = deserialize_banks(restored["positive_banks"])
        negative_banks = deserialize_banks(restored["negative_banks"])

    train_step_jit = jax.jit(
        train_step,
        static_argnames=(
            "num_classes",
            "ignore_index",
            "lambda_uncertainty",
            "ce_weight",
            "dice_weight",
            "drift_weight",
            "bank_ready",
            "drift_r_list",
            "drift_class_ids",
        ),
    )
    eval_step_jit = jax.jit(eval_step)

    wandb_run = maybe_init_wandb(cfg, output_dir, cfg["wandb_enabled"])
    save_every = int(cfg["train"]["save_every_epochs"])
    eval_every = int(cfg["eval"]["eval_every_epochs"])
    total_epochs = int(cfg["train"]["epochs"])

    print("[cloudseg-drift] compiling first train step; this can take a while...", flush=True)
    history = []
    first_train_compile_timed = False
    for epoch in range(start_epoch, total_epochs):
        print(f"[cloudseg-drift] epoch {epoch + 1}/{total_epochs}", flush=True)
        compile_start = None
        if not first_train_compile_timed:
            compile_start = time.perf_counter()
        state, train_metrics = run_train_epoch(
            train_loader,
            state,
            train_step_jit,
            split=f"train[{epoch}]",
            cfg=cfg,
            positive_banks=positive_banks,
            negative_banks=negative_banks,
        )
        if not first_train_compile_timed and compile_start is not None:
            print(
                f"[cloudseg-drift] first train epoch entered after JAX compile in {time.perf_counter() - compile_start:.2f}s",
                flush=True,
            )
            first_train_compile_timed = True

        train_summary = summarize_metrics(train_metrics, train_ds.class_names)
        train_main, train_aux = split_summary_metrics(train_summary)
        record = {
            "epoch": epoch,
            "train": {
                "summary": train_main,
                "aux": train_aux,
            },
        }
        record["train"]["aux"]["loss"] = float(train_metrics.get("loss", record["train"]["aux"].get("loss", 0.0)))
        record["train"]["aux"]["loss_seg"] = float(train_metrics.get("loss_seg", record["train"]["aux"].get("loss_seg", 0.0)))
        record["train"]["aux"]["loss_drift"] = float(train_metrics.get("loss_drift", record["train"]["aux"].get("loss_drift", 0.0)))
        drift_contrib = float(cfg["drift"]["weight"]) * metric_value(record["train"]["aux"], "loss_drift")
        total_loss = metric_value(record["train"]["aux"], "loss")
        drift_ratio = drift_contrib / max(total_loss, 1e-8)
        record["train"]["aux"]["drift_contrib"] = drift_contrib
        record["train"]["aux"]["drift_ratio"] = drift_ratio
        record["train"]["aux"]["lr"] = float(create_learning_rate_schedule(cfg, len(train_loader))(state.step))

        if (epoch + 1) % eval_every == 0:
            val_metrics = run_eval_epoch(val_loader, state, eval_step_jit, split=f"val[{epoch}]", cfg=cfg)
            val_summary = summarize_metrics(val_metrics, train_ds.class_names)
            val_main, val_aux = split_summary_metrics(val_summary)
            record["val"] = {
                "summary": val_main,
                "aux": val_aux,
            }
            write_json(output_paths["metric"] / f"epoch_{epoch:04d}.json", record)
            if val_metrics["mean_iou"] > best_miou:
                best_miou = float(val_metrics["mean_iou"])
                save_named_checkpoint(
                    output_paths["ckpt"] / "best.ckpt",
                    state,
                    epoch,
                    best_miou,
                    metadata,
                    cfg,
                    positive_banks,
                    negative_banks,
                )
                write_json(
                    output_paths["metric"] / "best.json",
                    {
                        "best_epoch": epoch,
                        "criterion": "mIoU",
                        "best_mIoU": float(record["val"]["summary"]["mIoU"]),
                        "train": record["train"],
                        "val": record["val"],
                    },
                )

        history.append(record)
        if "val" in record:
            print(
                f"[cloudseg-drift] epoch {epoch + 1} val mIoU={record['val']['summary']['mIoU']:.4f} "
                f"pixel_acc={record['val']['summary']['pixel_acc']:.4f} "
                f"drift={metric_value(record['train']['aux'], 'loss_drift'):.4f} "
                f"ratio={metric_value(record['train']['aux'], 'drift_ratio'):.3f}",
                flush=True,
            )
        else:
            print(
                f"[cloudseg-drift] epoch {epoch + 1} train loss={metric_value(record['train']['aux'], 'loss'):.4f} "
                f"drift={metric_value(record['train']['aux'], 'loss_drift'):.4f} "
                f"ratio={metric_value(record['train']['aux'], 'drift_ratio'):.3f} "
                f"lr={metric_value(record['train']['aux'], 'lr'):.6e}",
                flush=True,
            )

        save_named_checkpoint(
            output_paths["ckpt"] / "latest.ckpt",
            state,
            epoch,
            best_miou,
            metadata,
            cfg,
            positive_banks,
            negative_banks,
        )
        if (epoch + 1) % save_every == 0:
            save_named_checkpoint(
                output_paths["ckpt"] / f"epoch_{epoch:04d}.ckpt",
                state,
                epoch,
                best_miou,
                metadata,
                cfg,
                positive_banks,
                negative_banks,
            )

        if wandb_run is not None:
            payload = {"epoch": epoch}
            payload.update(flatten_metrics("train", record["train"]))
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
