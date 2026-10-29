"""Бинарные маски «бактерии / фон» на основе метода Оцу и вспомогательная очистка масок.

Важно: при окраске по Граму бактерии ТЁМНЫЕ (фиолетовые/красные) на СВЕТЛОМ фоне, поэтому
бактерии — пиксели НИЖЕ порога яркости (в исходных step10/step12 было наоборот, и маска выделяла
фон — исправлено). Канал для порога задаётся ``config.MASK_CHANNEL`` (по умолчанию — яркость
после выравнивания фона, см. stains.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class OtsuResult:
    """Результат бинаризации Оцу.

    Attributes:
        mask: bool (H, W), True — бактерии (тёмные пиксели).
        threshold: найденный порог Оцу (0..255): яркость для "gray"/"flat_gray", плотность для "od".
        class_gap: разность средних «тёмности» объектов и фона. Малое значение означает,
            что изображение однородное (бактерий, скорее всего, нет).
        is_empty: True, если ``class_gap`` < порога — маска в этом случае обнулена.
    """

    mask: np.ndarray
    threshold: float
    class_gap: float
    is_empty: bool


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """BGR -> оттенки серого (uint8). Если на вход уже подано серое изображение — вернуть как есть."""
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def darkness_image(img_bgr: np.ndarray, channel: str = "gray") -> np.ndarray:
    """Канал «насколько пиксель похож на окрашенную клетку» (uint8, больше = темнее/плотнее).

    ``channel``: "gray" — 255 − яркость; "flat_gray" — то же после выравнивания фона;
    "od" — суммарная оптическая плотность красителей (см. stains.py).
    """
    if img_bgr.ndim == 2:  # уже серое изображение
        return 255 - img_bgr
    if channel == "gray":
        return 255 - to_gray(img_bgr)
    if channel == "flat_gray":
        from .stains import flatten_background
        return 255 - to_gray(flatten_background(img_bgr))
    if channel == "od":
        from .stains import stain_density_image
        return stain_density_image(img_bgr)
    raise ValueError(f"Неизвестный канал маски: {channel!r}")


def otsu_bacteria_mask(
    img_bgr: np.ndarray,
    blur_ksize: int | None = None,
    min_class_gap: float | None = None,
    min_object_area: int | None = None,
    threshold: float | None = None,
    channel: str | None = None,
) -> OtsuResult:
    """Маска бактерий методом Оцу по каналу «тёмности» (окрашенные клетки на светлом фоне).

    Args:
        img_bgr: цветное изображение BGR uint8 (или серое).
        blur_ksize: ядро гауссова размытия перед порогом (0 — без размытия).
        min_class_gap: если средние классов отличаются меньше — считаем, что бактерий нет.
            Передайте 0, чтобы отключить проверку.
        min_object_area: компоненты меньше этой площади удаляются как пыль (0 — не удалять).
        threshold: готовый порог (например, посчитанный по всему снимку, а не по тайлу).
            Если None — порог Оцу вычисляется по этому изображению. Порог — в единицах ЯРКОСТИ
            для "gray"/"flat_gray" (как в курсовой) и в единицах канала плотности для "od".
        channel: канал (см. ``darkness_image``).
    Параметры со значением None берутся из ``config`` В МОМЕНТ ВЫЗОВА (OTSU_BLUR_KSIZE,
    OTSU_MIN_CLASS_GAP, OTSU_MIN_OBJECT_AREA, MASK_CHANNEL) — так их можно менять во время работы
    (например, в скриптах сравнения версий алгоритма).
    """
    blur_ksize = config.OTSU_BLUR_KSIZE if blur_ksize is None else blur_ksize
    min_class_gap = config.OTSU_MIN_CLASS_GAP if min_class_gap is None else min_class_gap
    min_object_area = config.OTSU_MIN_OBJECT_AREA if min_object_area is None else min_object_area
    channel = config.MASK_CHANNEL if channel is None else channel
    # Для яркостных каналов Оцу считается по самой яркости (порог в единицах яркости, бактерии —
    # пиксели ≤ порога), как в курсовой: так признак otsu_thresh и маска совпадают бит в бит с обученной
    # моделью. Для "od" — по плотности красителя (бактерии — пиксели > порога).
    as_brightness = channel != "od" or img_bgr.ndim == 2
    work = 255 - darkness_image(img_bgr, channel) if as_brightness else darkness_image(img_bgr, channel)
    if blur_ksize and blur_ksize > 1:
        work = cv2.GaussianBlur(work, (blur_ksize, blur_ksize), 0)
    if threshold is None:
        threshold, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(threshold)
    mask = work <= threshold if as_brightness else work > threshold
    dark = 255 - work.astype(np.int16) if as_brightness else work  # «тёмность» для оценки разрыва классов

    # Проверка «есть ли что делить»: расстояние между средними двух классов.
    if mask.any() and (~mask).any():
        class_gap = float(dark[mask].mean() - dark[~mask].mean())
    else:
        class_gap = 0.0
    is_empty = class_gap < min_class_gap
    if is_empty:
        mask = np.zeros_like(mask)
    elif min_object_area > 0:
        mask = remove_small_components(mask, min_object_area)

    return OtsuResult(mask=mask, threshold=threshold, class_gap=class_gap, is_empty=is_empty)


def otsu_threshold_and_gap(img_bgr: np.ndarray, blur_ksize: int = config.OTSU_BLUR_KSIZE) -> tuple[float, float]:
    """Только порог Оцу и разрыв между классами (без построения финальной маски)."""
    res = otsu_bacteria_mask(img_bgr, blur_ksize=blur_ksize, min_class_gap=0, min_object_area=0)
    return res.threshold, res.class_gap


def remove_small_components(mask: np.ndarray, min_area: int, connectivity: int = 8) -> np.ndarray:
    """Удалить из bool-маски связные компоненты площадью меньше ``min_area`` пикселей."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    if n <= 1:
        return mask.astype(bool)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # метка 0 — фон
    return keep[labels]
