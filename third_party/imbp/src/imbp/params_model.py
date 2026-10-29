"""Предсказание параметров SLIC (K — число суперпикселей, M — компактность) по признакам изображения.

Основной способ — «IoU-осознанный» Random Forest (развитие раздела 4.1 курсовой).
    В курсовой задача ставилась как классификация: два RandomForestClassifier угадывали точные
    оптимальные K и M. Но, как отмечено там же, соседние значения (например, K=750 и K=1000) дают
    почти одинаковое качество, а классификатор считает такой промах полной ошибкой. В итоге
    классификаторы проигрывали даже фиксированной паре K=1000, M=45 (см. reports/evaluation.md).

    Здесь один ``RandomForestRegressor`` учится предсказывать ОЖИДАЕМЫЙ IoU по входу
    [6 признаков изображения, K, M]. При инференсе модель оценивает все 36 пар сетки
    ``config.K_VALUES × config.M_VALUES`` и выбирает пару с максимальным предсказанным IoU.
    Промах «на соседнюю клетку» почти ничего не стоит, а дорогие ошибки модель избегает.

Запасной способ — аналитическая формула из прошлогодней работы (линейно от плотности Оцу).
Используется, если файла модели нет (например, при первом запуске до обучения).

Для интегратора:
    predictor = SlicParamPredictor.load()          # один раз при старте ПО
    k, m = predictor.predict(img_bgr)              # для каждого тайла/патча (≈404×405)
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from . import config
from .features import extract_features

# Версия формата признаков и модели. Увеличивать при любом изменении features.extract_features
# или входа модели — тогда старый файл модели не загрузится молча с несовместимыми данными.
FEATURES_VERSION = 3

# Имена столбцов входа регрессора: признаки изображения + кандидатные параметры SLIC.
MODEL_INPUTS = (*config.FEATURE_NAMES, "k", "m")


def analytical_params(img_bgr: np.ndarray, density: float | None = None) -> tuple[int, float]:
    """Аналитический расчёт (K, M) по плотности бактерий (метод прошлого года, ML/step10).

    K растёт линейно с плотностью (базово 2% от числа пикселей), M линейно убывает.
    Используется вместе со SLICO (``slic_zero=True``), как в исходной реализации.
    """
    if density is None:
        density = extract_features(img_bgr)["density"]
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    base_segments = int(total_pixels * 0.02)
    k = int(base_segments + base_segments * 3 * density)
    m = float(np.clip(10 * (1.0 - 0.6 * density), 5, 40))
    return k, m


def param_grid() -> np.ndarray:
    """Все пары (K, M) сетки, массив (36, 2)."""
    return np.array(list(itertools.product(config.K_VALUES, config.M_VALUES)), dtype=float)


@dataclass
class SlicParamPredictor:
    """IoU-осознанный выбор (K, M): RF-регрессор «признаки + (K, M) -> ожидаемый IoU».

    Attributes:
        model: обученный ``RandomForestRegressor`` (вход — ``MODEL_INPUTS``).
        grid: (n, 2) — кандидатные пары (K, M), среди которых делается выбор.
        meta: служебная информация (версия признаков, метрики обучения, версия sklearn).
    """

    model: RandomForestRegressor | None = None
    grid: np.ndarray = field(default_factory=param_grid)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ обучение
    @staticmethod
    def make_regressor(**params) -> RandomForestRegressor:
        """Регрессор с гиперпараметрами по умолчанию (подобраны кросс-валидацией в scripts/05_train.py)."""
        defaults = dict(n_estimators=300, min_samples_leaf=10, random_state=config.RANDOM_STATE, n_jobs=-1)
        defaults.update(params)
        return RandomForestRegressor(**defaults)

    @staticmethod
    def long_table(features: pd.DataFrame, iou_grid: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
        """Обучающая таблица «одна строка = (изображение, K, M)».

        Args:
            features: DataFrame со столбцами ``filename`` + ``config.FEATURE_NAMES``.
            iou_grid: таблица artifacts/iou_grid.csv (filename, k, m, iou_*).
            target: столбец IoU, который учится предсказывать модель (``iou_oracle`` или ``iou_otsu``).
        """
        long = iou_grid.merge(features[["filename", *config.FEATURE_NAMES]], on="filename")
        return long[list(MODEL_INPUTS)], long[target]

    def fit(self, features: pd.DataFrame, iou_grid: pd.DataFrame, target: str = "iou_oracle",
            **rf_params) -> "SlicParamPredictor":
        """Обучить регрессор на изображениях из ``features`` по их строкам из ``iou_grid``."""
        X, y = self.long_table(features, iou_grid, target)
        self.model = self.make_regressor(**rf_params).fit(X.to_numpy(float), y.to_numpy(float))
        self.meta.update(
            model_type="iou_aware_rf_regressor",
            target=target,
            rf_params=self.model.get_params(),
            inputs=list(MODEL_INPUTS),
            features_version=FEATURES_VERSION,
            grid=self.grid.tolist(),
            slic_iterations=config.SLIC_ITERATIONS,
            tile_size=list(config.TILE_SIZE),
            sklearn_version=sklearn.__version__,
            n_images=int(features["filename"].nunique()),
        )
        return self

    # ------------------------------------------------------------------ сохранение / загрузка
    def save(self, path: str | Path = config.MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "grid": self.grid, "meta": self.meta}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path = config.MODEL_PATH) -> "SlicParamPredictor":
        """Загрузить модель. Бросает ``FileNotFoundError``/``ValueError`` при отсутствии или несовместимости."""
        data = joblib.load(Path(path))
        meta = data.get("meta", {})
        if meta.get("features_version") != FEATURES_VERSION:
            raise ValueError(
                f"Модель {path} обучена на признаках версии {meta.get('features_version')}, "
                f"а код ожидает версию {FEATURES_VERSION}. Переобучите: scripts/04_build_features.py, 05_train.py"
            )
        return cls(model=data["model"], grid=np.asarray(data["grid"], dtype=float), meta=meta)

    @classmethod
    def load_or_none(cls, path: str | Path = config.MODEL_PATH) -> "SlicParamPredictor | None":
        """Как ``load``, но возвращает None (с предупреждением) вместо исключения — удобно для GUI."""
        try:
            return cls.load(path)
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(f"Модель параметров SLIC недоступна ({exc}); используется аналитическая формула.",
                          stacklevel=2)
            return None

    # ------------------------------------------------------------------ инференс
    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def expected_iou(self, feature_rows: np.ndarray) -> np.ndarray:
        """Предсказанный IoU для каждой пары сетки: вход (n_images, 6) -> выход (n_images, n_grid)."""
        feature_rows = np.atleast_2d(np.asarray(feature_rows, dtype=float))
        n_img, n_grid = len(feature_rows), len(self.grid)
        X = np.hstack([np.repeat(feature_rows, n_grid, axis=0), np.tile(self.grid, (n_img, 1))])
        return self.model.predict(X).reshape(n_img, n_grid)

    def predict_batch(self, feature_rows: np.ndarray) -> np.ndarray:
        """(K, M) для нескольких изображений сразу: (n_images, 6) -> (n_images, 2)."""
        best = self.expected_iou(feature_rows).argmax(axis=1)
        return self.grid[best].astype(int)

    def predict_from_features(self, features: dict[str, float]) -> tuple[int, int]:
        row = [[features[name] for name in config.FEATURE_NAMES]]
        k, m = self.predict_batch(row)[0]
        return int(k), int(m)

    def predict(self, img_bgr: np.ndarray) -> tuple[int, int]:
        """(K, M) для изображения размера порядка ``config.TILE_SIZE``."""
        return self.predict_from_features(extract_features(img_bgr))
