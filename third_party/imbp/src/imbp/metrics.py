"""Метрики качества сегментации: MIoU (Mask IoU) и BIoU (Boundary IoU)."""

from __future__ import annotations

import cv2
import numpy as np


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    """Mask IoU двух бинарных масок. Если обе маски пустые — 1.0 (идеальное совпадение)."""
    pred = pred.astype(bool)
    true = true.astype(bool)
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, true).sum() / union)


def _inner_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """Полоса шириной ``width`` пикселей вдоль внутренней границы маски."""
    m = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    # Паддинг, чтобы край изображения не считался границей объекта.
    padded = cv2.copyMakeBorder(m, width, width, width, width, cv2.BORDER_REPLICATE)
    eroded = cv2.erode(padded, kernel)[width:-width, width:-width]
    return (m - eroded).astype(bool)


def boundary_iou(pred: np.ndarray, true: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """Boundary IoU (Cheng et al., 2021): IoU полос вдоль границ масок.

    Ширина полосы = ``dilation_ratio`` × диагональ изображения (не меньше 1 пикселя).
    """
    h, w = true.shape[:2]
    width = max(1, int(round(dilation_ratio * np.hypot(h, w))))
    return iou(_inner_boundary(pred, width), _inner_boundary(true, width))


def instance_scores(pred: np.ndarray, true: np.ndarray, iou_threshold: float = 0.5) -> dict[str, float]:
    """Качество разделения на отдельные клетки (карты меток int, 0 — фон).

    Клетки сопоставляются по IoU > ``iou_threshold`` (при пороге ≥ 0.5 сопоставление однозначно).
    Возвращает:
        tp, fp, fn          — найденные / лишние / пропущенные клетки;
        f1                  — 2·TP / (2·TP + FP + FN), F1 обнаружения клеток;
        pq                  — panoptic quality: сумма IoU сопоставленных пар / (TP + FP/2 + FN/2);
        merged              — эталонные клетки, «склеенные» с соседями (≥ 2 эталонные клетки покрыты
                              одной предсказанной больше чем на половину);
        split               — эталонные клетки, «разрезанные» (≥ 2 предсказанные метки покрывают
                              каждая ≥ 25% эталонной клетки);
        n_pred, n_true      — число клеток.
    """
    pred = pred.astype(np.int64)
    true = true.astype(np.int64)
    n_p, n_t = int(pred.max()), int(true.max())
    if n_p == 0 or n_t == 0:
        return dict(tp=0, fp=n_p, fn=n_t, f1=1.0 if n_p == n_t == 0 else 0.0, pq=1.0 if n_p == n_t == 0 else 0.0,
                    merged=0, split=0, n_pred=n_p, n_true=n_t)
    # Матрица пересечений (n_t+1) × (n_p+1) через bincount по парам меток.
    inter = np.bincount(true.ravel() * (n_p + 1) + pred.ravel(), minlength=(n_t + 1) * (n_p + 1)).reshape(n_t + 1, n_p + 1)
    area_t, area_p = inter.sum(axis=1), inter.sum(axis=0)
    inter = inter[1:, 1:].astype(float)
    union = area_t[1:, None] + area_p[None, 1:] - inter
    iou_m = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    matched = iou_m > iou_threshold
    tp = int(matched.sum())
    fp, fn = n_p - tp, n_t - tp
    cover_t = inter / np.maximum(area_t[1:, None], 1)   # доля эталонной клетки, покрытая предсказанной
    merged = int((cover_t > 0.5).sum(axis=0)[(cover_t > 0.5).sum(axis=0) >= 2].sum())
    split = int(((cover_t >= 0.25).sum(axis=1) >= 2).sum())
    return dict(tp=tp, fp=fp, fn=fn, f1=2 * tp / (2 * tp + fp + fn), pq=float(iou_m[matched].sum() / (tp + fp / 2 + fn / 2)),
                merged=merged, split=split, n_pred=n_p, n_true=n_t)
