"""Сегментация методом суперпикселей SLIC с параметрами, предсказанными Random Forest (раздел 4.1).

Конвейер ``segment_slic``:
    1. Снимок режется на тайлы размером примерно ``config.TILE_SIZE`` (размер обучающих патчей),
       т.к. RF предсказывает абсолютное K для патча такого размера. Снимок 1616×1216 -> 3×4 тайла.
    2. Для каждого тайла строится маска Оцу (тёмные объекты на светлом фоне). Если классы Оцу
       почти не различаются (``OTSU_MIN_CLASS_GAP``) — в тайле бактерий нет, тайл пропускается.
       Локальный порог по тайлу оказался точнее глобального порога по всему снимку
       (MIoU 0.48 против 0.37 на 800 размеченных патчах) — освещение по полю зрения неравномерное.
    3. Признаки тайла -> RF -> (K, M) -> SLIC.
    4. Голосование: суперпиксель считается бактерией, если > ``SUPERPIXEL_VOTE_THRESHOLD`` его
       пикселей попали в маску Оцу. Так суперпиксели «достраивают» маску до границ объектов
       и убирают одиночный шум.
    5. Метки суперпикселей всех тайлов сшиваются в одну карту с уникальными номерами.

Функция ``oracle_mask`` голосует по ЭТАЛОННОЙ маске (Cellpose) и нужна только для оценки
«потолка» качества сетки суперпикселей — именно так считались 79% / 69.9% в курсовой.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.segmentation import slic

from .. import config
from ..masks import otsu_bacteria_mask
from ..params_model import SlicParamPredictor, analytical_params


@dataclass
class TileInfo:
    """Описание одного тайла: координаты (y0, y1, x0, x1) и использованные параметры SLIC."""

    bbox: tuple[int, int, int, int]
    k: int
    m: float
    source: str  # "rf" | "analytical" | "fixed" | "default"


@dataclass
class SlicResult:
    """Результат SLIC-сегментации.

    Attributes:
        mask: bool (H, W) — пиксели бактерий.
        labels: int32 (H, W) — карта объектов для отображения/морфологии (0 — фон). Сразу после
            ``segment_slic`` это суперпиксели; ``Analyzer.segment`` заменяет её картой ОТДЕЛЬНЫХ
            КЛЕТОК (instances.split_instances), а суперпиксели переносит в ``superpixels``.
        superpixels: int32 (H, W) — номера суперпикселей, попавших в бактерии (для отладки).
        tiles: параметры SLIC по непустым тайлам (для отображения пользователю / логов).
        otsu_threshold: средний по тайлам порог Оцу (для информации).
        is_empty: True, если ни в одном тайле по критерию Оцу бактерий не найдено.
    """

    mask: np.ndarray
    labels: np.ndarray
    superpixels: np.ndarray | None = None
    tiles: list[TileInfo] = field(default_factory=list)
    otsu_threshold: float = 0.0
    is_empty: bool = False
    n_rejected: int = 0  # сколько компонент удалил фильтр ложных объектов (заполняет pipeline)

    def describe(self) -> str:
        """Короткое описание для строки статуса GUI/лога."""
        if self.is_empty:
            return "SLIC: бактерии не обнаружены"
        suffix = f"; фильтр убрал {self.n_rejected} объект(ов)" if self.n_rejected else ""
        return self._describe_params() + suffix

    def _describe_params(self) -> str:
        ks = sorted({t.k for t in self.tiles})
        ms = sorted({t.m for t in self.tiles})
        src = ", ".join(sorted({t.source for t in self.tiles}))
        if len(self.tiles) == 1:
            return f"SLIC ({src}): K={ks[0]}, M={ms[0]:g}"
        return f"SLIC ({src}): {len(self.tiles)} тайлов, K∈{ks}, M∈{[f'{m:g}' for m in ms]}"


# ---------------------------------------------------------------------------- базовые операции

def run_slic(img_bgr: np.ndarray, k: int, m: float, iterations: int = config.SLIC_ITERATIONS,
             slic_zero: bool = False) -> np.ndarray:
    """Запустить SLIC. Возвращает карту суперпикселей int (H, W), номера с 1."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # SLIC работает в Lab, конвертирует из RGB
    return slic(img_rgb, n_segments=int(k), compactness=float(m), max_num_iter=iterations,
                start_label=1, slic_zero=slic_zero)


def vote_superpixels(segments: np.ndarray, reference: np.ndarray,
                     threshold: float = config.SUPERPIXEL_VOTE_THRESHOLD) -> np.ndarray:
    """Собрать bool-маску из суперпикселей, в которых доля ``reference`` > ``threshold``.

    Векторизованная замена цикла по ``np.unique(segments)`` из исходного кода (в десятки раз быстрее).
    """
    seg = segments.ravel()
    total = np.bincount(seg)
    hits = np.bincount(seg, weights=reference.ravel().astype(float), minlength=len(total))
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(total > 0, hits / np.maximum(total, 1), 0.0)
    return (fraction > threshold)[segments]


def oracle_mask(segments: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
    """«Идеальная» сборка маски из суперпикселей по эталону (только для оценки потолка сетки)."""
    return vote_superpixels(segments, true_mask, config.SUPERPIXEL_VOTE_THRESHOLD)


def tile_grid(shape: tuple[int, ...], tile_size: tuple[int, int] = config.TILE_SIZE) -> list[tuple[int, int, int, int]]:
    """Разбить изображение формы ``shape`` на почти равные тайлы размера ≈ ``tile_size``.

    Число строк/столбцов = round(H / th), round(W / tw) (минимум 1), поэтому снимок 1616×1216
    режется ровно на 3×4 — так же, как при подготовке обучающих патчей.
    """
    h, w = shape[:2]
    th, tw = tile_size
    n_rows = max(1, int(round(h / th)))
    n_cols = max(1, int(round(w / tw)))
    ys = np.linspace(0, h, n_rows + 1).astype(int)
    xs = np.linspace(0, w, n_cols + 1).astype(int)
    # int(...) — чтобы координаты были обычными int Python (сериализуются в JSON).
    return [(int(ys[r]), int(ys[r + 1]), int(xs[c]), int(xs[c + 1])) for r in range(n_rows) for c in range(n_cols)]


# ---------------------------------------------------------------------------- основной метод

def choose_params(tile_bgr: np.ndarray, predictor: SlicParamPredictor | None,
                  fixed: tuple[int, float] | None = None, use_analytical: bool = False) -> tuple[int, float, str]:
    """Выбрать (K, M) для тайла: фиксированные -> аналитика -> RF -> значения по умолчанию."""
    if fixed is not None:
        return int(fixed[0]), float(fixed[1]), "fixed"
    if use_analytical:
        k, m = analytical_params(tile_bgr)
        return k, m, "analytical"
    if predictor is not None and predictor.is_trained:
        k, m = predictor.predict(tile_bgr)
        return k, float(m), "rf"
    return config.DEFAULT_K, float(config.DEFAULT_M), "default"


def segment_slic(
    img_bgr: np.ndarray,
    predictor: SlicParamPredictor | None = None,
    fixed_params: tuple[int, float] | None = None,
    use_analytical: bool = False,
    tile_size: tuple[int, int] | None = config.TILE_SIZE,
    otsu_threshold: float | None = None,
) -> SlicResult:
    """Полный конвейер SLIC-сегментации (см. описание модуля).

    Args:
        img_bgr: снимок BGR uint8 любого размера.
        predictor: обученный ``SlicParamPredictor``. Если None и ``fixed_params``/``use_analytical``
            не заданы — используются ``config.DEFAULT_K/M``.
        fixed_params: принудительно заданные (K, M) для всех тайлов (ручная настройка).
        use_analytical: использовать аналитическую формулу прошлого года вместо RF.
        tile_size: размер тайла; None — не резать (весь снимок одним куском).
        otsu_threshold: единый порог Оцу для всех тайлов (None — свой порог в каждом тайле).
    """
    h, w = img_bgr.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    result = SlicResult(mask=np.zeros((h, w), bool), labels=labels)

    boxes = tile_grid(img_bgr.shape, tile_size) if tile_size else [(0, h, 0, w)]
    offset = 0
    thresholds = []
    for y0, y1, x0, x1 in boxes:
        tile = img_bgr[y0:y1, x0:x1]
        otsu = otsu_bacteria_mask(tile, threshold=otsu_threshold)
        thresholds.append(otsu.threshold)
        if otsu.is_empty or not otsu.mask.any():
            continue  # в тайле нет тёмных объектов — SLIC не нужен
        k, m, source = choose_params(tile, predictor, fixed_params, use_analytical)
        result.tiles.append(TileInfo((y0, y1, x0, x1), k, m, source))
        segments = run_slic(tile, k, m, slic_zero=(source == "analytical"))
        tile_mask = vote_superpixels(segments, otsu.mask)
        result.mask[y0:y1, x0:x1] = tile_mask
        labels[y0:y1, x0:x1] = np.where(tile_mask, segments + offset, 0)
        offset += int(segments.max())

    result.otsu_threshold = float(np.mean(thresholds))
    result.is_empty = not result.tiles
    return result
