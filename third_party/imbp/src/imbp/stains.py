"""Выравнивание фона и оптическая плотность (OD) красителей.

Зачем:
    * Освещение микроскопа неравномерное, а на мазке бывают размытые пятна красителя. Глобальный
      порог по яркости на таких снимках «заливает» пятна как бактерии. Если поделить снимок на
      оценку фона, остаются только локальные тёмные объекты (канал маски "flat_gray").
    * Закон Бугера–Ламберта–Бера: оптическая плотность OD = −log10(I / I_фон) пропорциональна
      количеству красителя. Кристаллический фиолетовый (Грам+) поглощает оранжево-красный свет,
      сафранин/фуксин (Грам−) — нет, поэтому доля OD в красном канале — признак окраски по Граму
      (morphology.gram_from_od).

Проверено и отвергнуто: деконволюция на два «физических» вектора красителей (метод Macenko).
Направления OD у культур одного типа окраски сильно различаются между препаратами (баланс белого,
интенсивность окраски), устойчивых векторов получить не удалось.

Все функции принимают BGR uint8 (как OpenCV).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def estimate_background(img_bgr: np.ndarray, scale: int = config.BACKGROUND_SCALE,
                        kernel_px: int = config.BACKGROUND_KERNEL) -> np.ndarray:
    """Оценка фона (float32, H×W×3): морфологическое закрытие убирает тёмные объекты, затем размытие.

    Считается на уменьшенной в ``scale`` раз копии (в десятки раз быстрее, фон всё равно гладкий).
    ``kernel_px`` — размер ядра в пикселях ИСХОДНОГО снимка; должен быть больше типичного скопления
    клеток, иначе крупное скопление само станет «фоном».
    """
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    k = max(3, (kernel_px // scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def closing(x: np.ndarray) -> np.ndarray:
        # Закрытие: тёмные объекты меньше ядра исчезают. Лёгкое размытие убирает «ступеньки».
        c = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)
        return cv2.GaussianBlur(c, (0, 0), sigmaX=max(1.0, k / 8), borderType=cv2.BORDER_REPLICATE)

    small = small.astype(np.float32)
    bg = closing(small)
    # У края кадра окно закрытия обрезано, и на сильном градиенте освещения фон завышается.
    # Второй проход по уже выровненному изображению (остаточный градиент мал) убирает эту ошибку.
    med = np.median(bg.reshape(-1, 3), axis=0)
    bg = bg * closing(small / np.maximum(bg, 1.0) * med) / med
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def flatten_background(img_bgr: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    """Снимок, поделённый на фон и приведённый к его медианной яркости (BGR uint8).

    После выравнивания фон однородный, а пятна/градиенты, которые крупнее ядра фона, исчезают.
    """
    bg = estimate_background(img_bgr) if background is None else background
    target = np.median(bg.reshape(-1, 3), axis=0)
    flat = img_bgr.astype(np.float32) / np.maximum(bg, 1.0) * target
    return np.clip(flat, 0, 255).astype(np.uint8)


def optical_density(img_bgr: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    """Оптическая плотность по каналам (float32, H×W×3, ≥ 0) относительно локального фона."""
    bg = estimate_background(img_bgr) if background is None else background
    od = -np.log10((img_bgr.astype(np.float32) + 1.0) / (bg + 1.0))
    return np.maximum(od, 0.0)


def stain_density_image(img_bgr: np.ndarray, od_max: float = config.OD_DISPLAY_MAX) -> np.ndarray:
    """Суммарная оптическая плотность красителей, отмасштабированная в uint8 (больше = больше красителя).

    Используется как канал для порога Оцу вместо яркости (см. ``masks.darkness_image``).
    """
    od = optical_density(img_bgr)
    total = od.sum(axis=2)
    return np.clip(total / od_max * 255.0, 0, 255).astype(np.uint8)
