"""Фильтр ложных объектов: Random Forest «клетка / артефакт» по признакам связной компоненты маски.

Зачем: анализ ошибок показал, что precision обоих методов ≈ 0.5 при recall ≈ 0.74 — методы находят
клетки, но вместе с ними пыль, пятна красителя, расфокусированные пятна. Отличить их по одному
порогу яркости нельзя, а по форме, контрасту с окружением и резкости — можно.

Как работает:
    1. маска любого метода (SLIC / GMM / Оцу) разбивается на связные компоненты;
    2. для каждой компоненты считаются признаки (``OBJECT_FEATURES``) — только локальные, не зависящие
       от размера снимка, поэтому модель, обученная на патчах, применима к целым снимкам;
    3. RF выдаёт вероятность «клетка»; компоненты с вероятностью < порога удаляются.
Обучение — scripts/07_train_object_filter.py (метка: ≥ 50% площади компоненты лежит в эталоне).

Для интегратора: фильтр включается флагом ``config.USE_OBJECT_FILTER`` и загружается автоматически
в ``Analyzer``; если файла модели нет — маска не фильтруется (с предупреждением).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels, find_boundaries
from sklearn.ensemble import RandomForestClassifier

from . import config
from .stains import estimate_background, optical_density

# Версия набора признаков: увеличить при изменении component_features (старая модель не загрузится).
OBJECT_FEATURES_VERSION = 1

OBJECT_FEATURES = (
    "log_area",          # log площади (пиксели)
    "solidity",          # площадь / площадь выпуклой оболочки (клетки выпуклые, пятна — рваные)
    "extent",            # площадь / площадь описывающего прямоугольника
    "eccentricity",      # вытянутость эллипса
    "minor_axis",        # толщина объекта (клетки примерно одинаковой толщины)
    "od_mean",           # средняя оптическая плотность внутри (насыщенность окраски)
    "od_max",            # максимум OD (у расфокуса максимум низкий)
    "contrast_ring",     # яркость кольца вокруг − яркость внутри (локальный контраст)
    "edge_gradient",     # средний градиент яркости на границе (резкость края; расфокус — мягкий край)
    "lap_var",           # дисперсия лапласиана внутри (текстура/резкость)
    "lab_a",             # Lab a* (зелёный ↔ пурпурный)
    "lab_b",             # Lab b* (синий ↔ жёлтый): фиолет/красный против жёлтых/бурых пятен
    "lab_l_std",         # неоднородность яркости внутри
)


def component_features(img_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Признаки всех связных компонент маски.

    Returns:
        labels: int32 (H, W) — карта компонент (0 — фон);
        features: DataFrame, строка i — компонента с меткой i+1, столбцы ``OBJECT_FEATURES``.
    """
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return labels.astype(np.int32), pd.DataFrame(columns=list(OBJECT_FEATURES))
    idx = np.arange(1, n + 1)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    od = optical_density(img_bgr, estimate_background(img_bgr)).sum(axis=2)

    props = regionprops_table(labels, properties=("area", "solidity", "extent", "eccentricity",
                                                  "axis_minor_length"))
    # Кольцо шириной 4 px вокруг объекта (без других объектов) — локальный фон.
    ring = expand_labels(labels, distance=4)
    ring[mask] = 0
    ring_mean = np.asarray(ndi.mean(gray, ring, idx))
    inside_mean = np.asarray(ndi.mean(gray, labels, idx))

    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    edge = np.where(find_boundaries(labels, mode="inner"), labels, 0)
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    feats = pd.DataFrame({
        "log_area": np.log(props["area"].astype(float)),
        "solidity": props["solidity"],
        "extent": props["extent"],
        "eccentricity": props["eccentricity"],
        "minor_axis": props["axis_minor_length"],
        "od_mean": ndi.mean(od, labels, idx),
        "od_max": ndi.maximum(od, labels, idx),
        "contrast_ring": np.nan_to_num(ring_mean - inside_mean),
        "edge_gradient": np.nan_to_num(np.asarray(ndi.mean(grad, edge, idx))),
        "lap_var": ndi.variance(lap, labels, idx),
        "lab_a": ndi.mean(lab[..., 1], labels, idx),
        "lab_b": ndi.mean(lab[..., 2], labels, idx),
        "lab_l_std": ndi.standard_deviation(lab[..., 0], labels, idx),
    })
    return labels.astype(np.int32), feats[list(OBJECT_FEATURES)].astype(float)


def component_targets(labels: np.ndarray, reference: np.ndarray, min_overlap: float = 0.5) -> np.ndarray:
    """Метки обучения: компонента — клетка (1), если ≥ ``min_overlap`` её площади лежит в эталоне."""
    n = int(labels.max())
    if n == 0:
        return np.zeros(0, int)
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    hit = np.bincount(labels.ravel(), weights=reference.ravel().astype(float), minlength=n + 1)[1:]
    return (hit / np.maximum(area, 1) >= min_overlap).astype(int)


@dataclass
class ObjectFilter:
    """RF-классификатор «клетка / артефакт» для компонент маски.

    Attributes:
        model: обученный ``RandomForestClassifier``.
        threshold: минимальная вероятность «клетка», чтобы компонента осталась в маске.
        meta: служебная информация (версия признаков, метрики, версия sklearn).
    """

    model: RandomForestClassifier | None = None
    threshold: float = 0.5
    meta: dict = field(default_factory=dict)

    @staticmethod
    def make_classifier(**params) -> RandomForestClassifier:
        defaults = dict(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                        random_state=config.RANDOM_STATE, n_jobs=-1)
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    def fit(self, X: pd.DataFrame, y: np.ndarray, **rf_params) -> "ObjectFilter":
        self.model = self.make_classifier(**rf_params).fit(X[list(OBJECT_FEATURES)].to_numpy(float), y)
        self.meta.update(features=list(OBJECT_FEATURES), features_version=OBJECT_FEATURES_VERSION,
                         sklearn_version=sklearn.__version__, n_objects=int(len(y)),
                         positive_share=float(np.mean(y)))
        return self

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(0)
        return self.model.predict_proba(feats[list(OBJECT_FEATURES)].to_numpy(float))[:, 1]

    def filter_mask(self, img_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Удалить из маски компоненты-артефакты. Возвращает (новая маска, вероятности по компонентам)."""
        labels, feats = component_features(img_bgr, mask)
        proba = self.predict_proba(feats)
        keep = np.zeros(len(proba) + 1, bool)
        keep[1:] = proba >= self.threshold
        return keep[labels], proba

    # ------------------------------------------------------------------ сохранение / загрузка
    def save(self, path: str | Path = config.OBJECT_FILTER_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold, "meta": self.meta}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path = config.OBJECT_FILTER_PATH) -> "ObjectFilter":
        data = joblib.load(Path(path))
        if data.get("meta", {}).get("features_version") != OBJECT_FEATURES_VERSION:
            raise ValueError(f"Фильтр {path} обучен на другой версии признаков — переобучите scripts/07_train_object_filter.py")
        return cls(model=data["model"], threshold=float(data["threshold"]), meta=data["meta"])

    @classmethod
    def load_or_none(cls, path: str | Path = config.OBJECT_FILTER_PATH) -> "ObjectFilter | None":
        try:
            return cls.load(path)
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(f"Фильтр ложных объектов недоступен ({exc}); маски не фильтруются.", stacklevel=2)
            return None
