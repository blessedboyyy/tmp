"""Разделение маски бактерий на ОТДЕЛЬНЫЕ клетки (инстансы).

Маска сегментации (SLIC / GMM / «Авто») отвечает на вопрос «где бактерии», но касающиеся клетки в ней
сливаются в одну компоненту, а SLIC вдобавок режет каждую клетку на много суперпикселей. Для подсчёта
кокков/бацилл и долей Грама нужна карта, где каждая клетка — одна метка. Её строит ``split_instances``.

Алгоритм (маркерный водораздел):
    1. Рельеф f = расстояние до границы маски (сглаженное) + β · «тёмность» (сглаженная яркость
       выровненного снимка, инвертированная). У окрашенных клеток центр плотнее края, поэтому тёмность
       помогает разделить сросшиеся кокки, у которых по форме почти нет перетяжки.
    2. Маркеры — h-максимумы рельефа: вершина считается отдельной клеткой, только если она выше седла,
       соединяющего её с соседней вершиной, хотя бы на h. В отличие от порога «доля от максимума
       компоненты» (было в морфологии этапа 1) это работает и в крупных скоплениях: каждая клетка
       сравнивается со своими соседями, а не с самой большой клеткой скопления.
    3. Водораздел по −f от маркеров внутри маски.
    4. Склейка пересегментации: соседние куски объединяются, если ширина их контакта больше
       ``neck_ratio`` × диаметр меньшего куска — значит, между ними нет перетяжки (например, палочку
       разрезало поперёк), это одна клетка.
Параметры подобраны по разметке отдельных клеток Omnipose (scripts/10_tune_instances.py) и лежат в config.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from . import config


def relief(img_bgr: np.ndarray, mask: np.ndarray, sigma: float, beta: float) -> np.ndarray:
    """Рельеф для водораздела: сглаженное расстояние до границы + β · сглаженная тёмность (float32)."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if sigma > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), sigma)
    if beta <= 0:
        return dist * mask
    from .masks import darkness_image  # локальный импорт: masks -> stains -> config

    dark = darkness_image(img_bgr, "flat_gray").astype(np.float32)
    dark = cv2.GaussianBlur(dark, (0, 0), max(sigma, 1.0))
    # Тёмность относительно фона снимка (медиана вне маски), в уровнях яркости.
    background = np.median(dark[~mask]) if (~mask).any() else 0.0
    return (dist + beta * np.maximum(dark - background, 0)) * mask


def merge_wide_contacts(labels: np.ndarray, dist: np.ndarray, neck_ratio: float) -> np.ndarray:
    """Склеить соседние метки, если их контакт шире ``neck_ratio`` × диаметр меньшей из них."""
    n = int(labels.max())
    if n < 2 or neck_ratio <= 0:
        return labels
    # Пары соседей по 4-связности и длина контакта (число пар пикселей на границе).
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], axis=1))
    pairs = np.concatenate(pairs)
    if len(pairs) == 0:
        return labels
    uniq, contact = np.unique(pairs, axis=0, return_counts=True)
    radius = np.zeros(n + 1)
    radius[1:] = ndi.maximum(dist, labels, np.arange(1, n + 1))

    parent = np.arange(n + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), length in zip(uniq, contact):
        # Контакт считается с обеих сторон границы — делим пополам.
        if length / 2 > neck_ratio * 2 * min(radius[a], radius[b]):
            parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(n + 1)])
    _, relabeled = np.unique(roots, return_inverse=True)
    return relabeled.reshape(-1)[labels].astype(np.int32)


def split_instances(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    h: float | None = None,
    sigma: float | None = None,
    beta: float | None = None,
    neck_ratio: float | None = None,
) -> np.ndarray:
    """Карта отдельных клеток int32 (H, W): 0 — фон, 1..N — клетки.

    Параметры None берутся из config (INSTANCE_H, INSTANCE_SIGMA, INSTANCE_BETA, INSTANCE_NECK_RATIO).
    """
    h = config.INSTANCE_H if h is None else h
    sigma = config.INSTANCE_SIGMA if sigma is None else sigma
    beta = config.INSTANCE_BETA if beta is None else beta
    neck_ratio = config.INSTANCE_NECK_RATIO if neck_ratio is None else neck_ratio

    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, np.int32)
    f = relief(img_bgr, mask, sigma, beta)
    markers, _ = ndi.label(h_maxima(f, h) & mask, structure=np.ones((3, 3)))
    # Компоненты маски без единого маркера (очень мелкие/плоские) — каждая становится одной клеткой.
    comp, n_comp = ndi.label(mask, structure=np.ones((3, 3)))
    has_marker = np.zeros(n_comp + 1, bool)
    has_marker[np.unique(comp[markers > 0])] = True
    orphan = mask & ~has_marker[comp]
    if orphan.any():
        markers = np.where(orphan, comp + markers.max(), markers)
    labels = watershed(-f, markers, mask=mask)
    _, labels = np.unique(labels, return_inverse=True)  # плотная перенумерация 0..N
    labels = labels.reshape(mask.shape).astype(np.int32)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return merge_wide_contacts(labels, dist, neck_ratio)
