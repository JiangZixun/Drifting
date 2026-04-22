from __future__ import annotations

import jax
import jax.numpy as jnp


def dice_loss(
    inputs: jax.Array,
    target: jax.Array,
    *,
    n_classes: int,
    ignore_index: int = 10,
    softmax: bool = False,
) -> jax.Array:
    if softmax:
        inputs = jax.nn.softmax(inputs, axis=1)

    mask = target != ignore_index
    target = target * mask.astype(target.dtype)
    target_one_hot = jax.nn.one_hot(target, n_classes, dtype=jnp.float32)
    target_one_hot = jnp.transpose(target_one_hot, (0, 3, 1, 2))
    inputs = inputs * mask[:, None, :, :]

    smooth = 1e-5
    loss = 0.0
    for i in range(n_classes):
        score = inputs[:, i]
        tgt = target_one_hot[:, i]
        intersect = jnp.sum(score * tgt)
        y_sum = jnp.sum(tgt * tgt)
        z_sum = jnp.sum(score * score)
        dice = 1.0 - ((2.0 * intersect + smooth) / (z_sum + y_sum + smooth))
        loss = loss + dice
    return loss / n_classes


def attention_weighted_ce_loss(
    logits: jax.Array,
    targets: jax.Array,
    *,
    num_classes: int,
    lambda_uncertainty: float = 0.5,
    ignore_index: int = 10,
) -> jax.Array:
    probs = jax.nn.softmax(logits, axis=1)
    log_probs = jnp.log(probs + 1e-8)
    entropy = -jnp.sum(probs * log_probs, axis=1)

    valid_mask = targets != ignore_index
    valid_targets = jnp.clip(targets, 0, num_classes - 1)
    total_valid = jnp.maximum(valid_mask.sum(), 1)

    w_base = []
    entropy_mean = []
    for c in range(num_classes):
        class_mask = jnp.logical_and(targets == c, valid_mask)
        class_count = class_mask.sum()
        base = jnp.where(class_count > 0, (total_valid - class_count) / total_valid, 0.0)
        mean_ent = jnp.where(class_count > 0, jnp.sum(entropy * class_mask) / class_count, 0.0)
        w_base.append(base)
        entropy_mean.append(mean_ent)
    w_base = jnp.asarray(w_base, dtype=jnp.float32)
    entropy_mean = jnp.asarray(entropy_mean, dtype=jnp.float32)
    w_combined = w_base * (1.0 + lambda_uncertainty * entropy_mean)

    weight_map = jnp.take(w_combined, valid_targets, axis=0)
    weight_map = weight_map * valid_mask

    logits_flat = jnp.transpose(logits, (0, 2, 3, 1)).reshape(-1, num_classes)
    targets_flat = valid_targets.reshape(-1)
    ce_loss = jax.nn.log_softmax(logits_flat, axis=-1)
    ce_loss = -jnp.take_along_axis(ce_loss, targets_flat[:, None], axis=1).reshape(valid_targets.shape)
    ce_loss = ce_loss * valid_mask
    return (ce_loss * weight_map).sum() / (valid_mask.sum() + 1e-6)


def focal_loss(
    logits: jax.Array,
    targets: jax.Array,
    *,
    num_classes: int,
    ignore_index: int = 10,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> jax.Array:
    valid_mask = targets != ignore_index
    valid_targets = jnp.clip(targets, 0, num_classes - 1)

    logits_flat = jnp.transpose(logits, (0, 2, 3, 1)).reshape(-1, num_classes)
    targets_flat = valid_targets.reshape(-1)
    valid_flat = valid_mask.reshape(-1).astype(jnp.float32)

    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    probs = jnp.exp(log_probs)
    true_log_probs = jnp.take_along_axis(log_probs, targets_flat[:, None], axis=1).squeeze(-1)
    true_probs = jnp.take_along_axis(probs, targets_flat[:, None], axis=1).squeeze(-1)

    focal_factor = jnp.power(1.0 - true_probs, gamma)
    loss = -alpha * focal_factor * true_log_probs
    loss = loss * valid_flat
    return loss.sum() / (valid_flat.sum() + 1e-6)


def cloudseg_loss(
    logits_bhwc: jax.Array,
    labels: jax.Array,
    *,
    num_classes: int,
    ignore_index: int = 10,
    primary_loss: str = "attention_ce",
    lambda_uncertainty: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = jnp.transpose(logits_bhwc, (0, 3, 1, 2))
    if primary_loss == "focal":
        loss1 = focal_loss(
            logits,
            labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    elif primary_loss == "attention_ce":
        loss1 = attention_weighted_ce_loss(
            logits,
            labels,
            num_classes=num_classes,
            lambda_uncertainty=lambda_uncertainty,
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unsupported primary_loss={primary_loss!r}; expected 'attention_ce' or 'focal'.")
    loss2 = dice_loss(
        logits,
        labels,
        n_classes=num_classes,
        ignore_index=ignore_index,
        softmax=True,
    )
    total = ce_weight * loss1 + dice_weight * loss2
    metrics = {
        "loss": total,
        "loss_ce": loss1,
        "loss_dice": loss2,
    }
    if primary_loss == "focal":
        metrics["loss_focal"] = loss1
    return total, metrics
