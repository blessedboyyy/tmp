"""Отрисовка результатов поверх исходного снимка (раздел 4.4).

Все функции принимают и возвращают BGR uint8 (как OpenCV); для Tkinter/PIL конвертируйте в RGB.
Цвета заданы в BGR.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

from .morphology import ARTIFACT, COCCUS, MorphologyResult

BOUNDARY_COLOR = (255, 255, 0)    # бирюзовый — хорошо виден на жёлтом/бежевом фоне
COCCUS_COLOR = (0, 255, 0)        # зелёный — кокки
BACILLUS_COLOR = (255, 0, 0)      # синий — бациллы
ARTIFACT_COLOR = (128, 128, 128)  # серый — артефакты (рисуются только при show_artifacts=True)


def draw_boundaries(img_bgr: np.ndarray, labels: np.ndarray, color=BOUNDARY_COLOR) -> np.ndarray:
    """Наложить границы областей карты ``labels`` (0 — фон).

    Заменяет приём ``cv2.Canny(labels*255/labels.max())`` из исходного кода: тот терял границы
    между соседними метками с близкими номерами и делил на ноль при пустой карте.
    """
    out = img_bgr.copy()
    if labels.any():
        out[find_boundaries(labels, mode="inner")] = color
    return out


def draw_objects(img_bgr: np.ndarray, morph: MorphologyResult, thickness: int = 2,
                 show_artifacts: bool = False) -> np.ndarray:
    """Нарисовать эллипсы вокруг бактерий: зелёный — кокки, синий — бациллы."""
    out = img_bgr.copy()
    for obj in morph.objects:
        if obj.morphotype == ARTIFACT:
            if show_artifacts:
                x, y, w, h = obj.bbox
                cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), ARTIFACT_COLOR, 1)
            continue
        color = COCCUS_COLOR if obj.morphotype == COCCUS else BACILLUS_COLOR
        cv2.ellipse(out, obj.ellipse, color, thickness)
    return out


def render_analysis(img_bgr: np.ndarray, labels: np.ndarray, morph: MorphologyResult | None) -> np.ndarray:
    """Итоговая картинка для пользователя: границы сегментации + (если есть) эллипсы объектов."""
    out = draw_boundaries(img_bgr, labels)
    if morph is not None:
        out = draw_objects(out, morph)
    return out
