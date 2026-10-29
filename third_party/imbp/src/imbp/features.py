"""Визуальные признаки изображения для предсказания параметров SLIC (раздел 4.1 курсовой).

ЕДИНАЯ функция для обучения и инференса. В исходном коде признаки при обучении (step4) и в GUI
(step9) считались по-разному (с инверсией маски / без, с размытием / без), из-за чего модель
получала на входе «чужое» распределение. Теперь и scripts/04_build_features.py, и ПО вызывают
только ``extract_features``.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import shannon_entropy

from . import config
from .masks import otsu_bacteria_mask, to_gray


def extract_features(img_bgr: np.ndarray) -> dict[str, float]:
    """Шесть признаков изображения (порядок — ``config.FEATURE_NAMES``).

    * brightness — средняя яркость серого изображения;
    * contrast — стандартное отклонение яркости;
    * entropy — энтропия Шеннона (насыщенность деталями);
    * otsu_thresh — порог бинаризации Оцу (по размытому изображению);
    * density — доля площади, занятая тёмными объектами по маске Оцу;
    * avg_size — средняя площадь связной компоненты маски Оцу (пиксели).

    Маска Оцу здесь берётся «сырой» (без проверки на пустоту и без удаления пыли), чтобы признаки
    описывали изображение как есть; решение «есть ли бактерии» принимается позже, при сегментации.
    """
    gray = to_gray(img_bgr)
    otsu = otsu_bacteria_mask(gray, min_class_gap=0, min_object_area=0)
    mask = otsu.mask

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "entropy": float(shannon_entropy(gray)),
        "otsu_thresh": otsu.threshold,
        "density": float(mask.mean()),
        "avg_size": float(areas.mean()) if len(areas) else 0.0,
    }


def features_to_row(features: dict[str, float]) -> np.ndarray:
    """Признаки -> вектор (1, n_features) в порядке ``config.FEATURE_NAMES`` (вход модели)."""
    return np.array([[features[name] for name in config.FEATURE_NAMES]], dtype=float)
