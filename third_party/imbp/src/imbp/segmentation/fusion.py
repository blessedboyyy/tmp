"""Режим «Авто»: для каждого тайла снимка автоматически выбирается лучший метод сегментации.

Зачем: методы ошибаются на разных снимках — SLIC «заливает» градиентные пятна красителя, GMM теряет
бледные клетки, маска Оцу иногда точнее обоих (см. reports/slic_vs_gmm.png). Простое объединение
масок SLIC ∪ GMM проверено и отвергнуто: оно хуже лучшего одиночного метода даже с фильтром
(52.3% против 54.8% на train).

Как работает «Авто» (``segment_auto``):
    1. снимок режется на тайлы размера обучающих патчей (как в SLIC);
    2. в каждом тайле строятся маски трёх кандидатов: Оцу (по выровненному фону), SLIC (RF-параметры), GMM;
    3. ``MethodChooser`` — RF-регрессор — по признакам тайла предсказывает IoU каждого кандидата,
       берётся маска с максимальным предсказанным IoU («IoU-осознанный» выбор, как для параметров SLIC);
    4. фильтр ложных объектов применяется к итоговой маске в pipeline.
Обучение — scripts/08_train_fusion.py. Без файла модели «Авто» использует маску Оцу (лучший одиночный
метод на train).
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from .. import config
from ..features import extract_features
from ..masks import otsu_bacteria_mask
from ..metrics import iou
from .gmm import segment_gmm
from .slic import segment_slic, tile_grid

CHOOSER_METHODS = ("otsu", "slic", "gmm")
CHOOSER_FEATURES_VERSION = 1
DEFAULT_METHOD = "otsu"  # если модели выбора нет


def candidate_masks(tile_bgr: np.ndarray, predictor=None) -> dict[str, np.ndarray]:
    """Маски трёх кандидатов для одного тайла/патча (без фильтра ложных объектов)."""
    return {
        "otsu": otsu_bacteria_mask(tile_bgr).mask,
        "slic": segment_slic(tile_bgr, predictor=predictor, use_analytical=predictor is None, tile_size=None).mask,
        "gmm": segment_gmm(tile_bgr).mask,
    }


def chooser_features(tile_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Признаки тайла для выбора метода: признаки курсовой + статистики масок кандидатов и их согласие."""
    feats = dict(extract_features(tile_bgr))
    for m in CHOOSER_METHODS:
        feats[f"{m}_frac"] = float(masks[m].mean())
        feats[f"{m}_ncomp"] = float(cv2.connectedComponents(masks[m].astype(np.uint8))[0] - 1)
    for a, b in (("otsu", "slic"), ("otsu", "gmm"), ("slic", "gmm")):
        feats[f"iou_{a}_{b}"] = iou(masks[a], masks[b])
    return feats


@dataclass
class MethodChooser:
    """RF-регрессор: признаки тайла -> ожидаемый IoU каждого метода из ``CHOOSER_METHODS``."""

    model: RandomForestRegressor | None = None
    feature_names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def fit(self, feats: pd.DataFrame, iou_per_method: pd.DataFrame, **rf_params) -> "MethodChooser":
        params = dict(n_estimators=300, min_samples_leaf=10, random_state=config.RANDOM_STATE, n_jobs=-1)
        params.update(rf_params)
        self.feature_names = list(feats.columns)
        self.model = RandomForestRegressor(**params).fit(
            feats.to_numpy(float), iou_per_method[list(CHOOSER_METHODS)].to_numpy(float))
        self.meta.update(methods=list(CHOOSER_METHODS), features=self.feature_names,
                         features_version=CHOOSER_FEATURES_VERSION, sklearn_version=sklearn.__version__)
        return self

    def choose(self, feats: pd.DataFrame) -> np.ndarray:
        """Название лучшего метода для каждой строки признаков."""
        pred = self.model.predict(feats[self.feature_names].to_numpy(float))
        return np.array(CHOOSER_METHODS)[np.atleast_2d(pred).argmax(axis=1)]

    def save(self, path: str | Path = config.METHOD_CHOOSER_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names, "meta": self.meta}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path = config.METHOD_CHOOSER_PATH) -> "MethodChooser":
        data = joblib.load(Path(path))
        if data.get("meta", {}).get("features_version") != CHOOSER_FEATURES_VERSION:
            raise ValueError(f"Модель выбора {path} несовместима — переобучите scripts/08_train_fusion.py")
        return cls(model=data["model"], feature_names=list(data["features"]), meta=data["meta"])

    @classmethod
    def load_or_none(cls, path: str | Path = config.METHOD_CHOOSER_PATH) -> "MethodChooser | None":
        try:
            return cls.load(path)
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(f"Модель выбора метода недоступна ({exc}); «Авто» использует маску Оцу.", stacklevel=2)
            return None


@dataclass
class AutoResult:
    """Результат режима «Авто».

    Attributes:
        mask: bool (H, W) — итоговая маска (после фильтра, если он применён в pipeline).
        labels: int32 (H, W) — связные компоненты маски (0 — фон); ``Analyzer.segment`` заменяет их
            картой отдельных клеток (instances.split_instances).
        tile_methods: [(bbox (y0, y1, x0, x1), выбранный метод)] по тайлам.
        n_rejected: сколько компонент удалил фильтр ложных объектов.
    """

    mask: np.ndarray
    labels: np.ndarray
    tile_methods: list[tuple[tuple[int, int, int, int], str]] = field(default_factory=list)
    n_rejected: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.mask.any()

    def describe(self) -> str:
        counts = pd.Series([m for _, m in self.tile_methods]).value_counts()
        names = {"otsu": "Оцу", "slic": "SLIC", "gmm": "GMM"}
        text = "Авто: " + ", ".join(f"{names[m]} — {n} тайл(ов)" for m, n in counts.items())
        if self.n_rejected:
            text += f"; фильтр убрал {self.n_rejected} объект(ов)"
        return text


def segment_auto(img_bgr: np.ndarray, predictor=None, chooser: MethodChooser | None = None,
                 tile_size: tuple[int, int] | None = config.TILE_SIZE) -> AutoResult:
    """Выбор метода по тайлам (см. описание модуля). Фильтр ложных объектов применяется в pipeline."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), bool)
    result = AutoResult(mask=mask, labels=np.zeros((h, w), np.int32))
    boxes = tile_grid(img_bgr.shape, tile_size) if tile_size else [(0, h, 0, w)]

    def process(box: tuple[int, int, int, int]) -> tuple[str, np.ndarray]:
        y0, y1, x0, x1 = box
        tile = img_bgr[y0:y1, x0:x1]
        masks = candidate_masks(tile, predictor)
        if chooser is None or not any(m.any() for m in masks.values()):
            method = DEFAULT_METHOD
        else:
            method = str(chooser.choose(pd.DataFrame([chooser_features(tile, masks)]))[0])
        return method, masks[method]

    # Тайлы независимы: считаем их в потоках (OpenCV, numpy и skimage отпускают GIL) — на снимке
    # 1616×1216 (12 тайлов) это в несколько раз быстрее последовательного прохода.
    with ThreadPoolExecutor(max_workers=min(len(boxes), os.cpu_count() or 1)) as pool:
        outputs = list(pool.map(process, boxes))
    for (y0, y1, x0, x1), (method, tile_mask) in zip(boxes, outputs):
        mask[y0:y1, x0:x1] = tile_mask
        result.tile_methods.append(((y0, y1, x0, x1), method))
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    result.labels = labels.astype(np.int32)
    return result
