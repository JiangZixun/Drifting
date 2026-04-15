from __future__ import annotations

import numpy as np


def confusion_matrix(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes)
    conf_mat = np.bincount(
        num_classes * label[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return conf_mat[:num_classes, :num_classes]


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def evaluate(conf_mat):
    matrix = conf_mat
    total = matrix.sum()
    acc = float(np.diag(matrix).sum() / total) if total > 0 else 0.0
    acc_per_class = _safe_divide(np.diag(matrix), matrix.sum(axis=1))
    pre = float(np.mean(acc_per_class))

    recall_class = _safe_divide(np.diag(matrix), matrix.sum(axis=0))
    recall = float(np.mean(recall_class))
    f1_score = float((2 * pre * recall) / (pre + recall)) if (pre + recall) > 0 else 0.0

    iou = _safe_divide(np.diag(matrix), matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix))
    mean_iou = float(np.mean(iou))

    pe = float(np.dot(np.sum(matrix, axis=0), np.sum(matrix, axis=1)) / (total ** 2)) if total > 0 else 0.0
    kappa = float((acc - pe) / (1 - pe)) if pe != 1.0 else 0.0
    return {
        "acc": acc,
        "acc_per_class": acc_per_class,
        "pre": pre,
        "iou": iou,
        "mean_iou": mean_iou,
        "kappa": kappa,
        "f1_score": f1_score,
        "recall": recall,
    }
