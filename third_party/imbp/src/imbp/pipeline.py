"""Высокоуровневый конвейер: снимок -> сегментация -> морфология -> статистика + картинка.

Главная точка входа для интеграции в ПО::

    from imbp import Analyzer, load_image

    analyzer = Analyzer()                          # грузит модель RF один раз
    img = load_image("снимок.bmp")
    result = analyzer.analyze(img, method="auto")  # "auto" (выбор метода по тайлам), "slic" или "gmm"
    result.summary.text()                          # статистика для пользователя
    result.overlay                                 # BGR-картинка с границами и эллипсами
    result.to_dict()                               # всё в JSON-совместимом виде

Тот же конвейер по шагам (как в GUI: сначала сегментация, потом анализ):
``analyzer.segment(img, method)`` -> ``analyzer.classify(img, segmentation)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from . import config
from .instances import split_instances
from .morphology import MorphologyResult, MorphologySummary, classify_objects
from .object_filter import ObjectFilter
from .params_model import SlicParamPredictor
from .segmentation import AutoResult, GmmResult, MethodChooser, SlicResult, segment_auto, segment_gmm, segment_slic
from .visualization import render_analysis

Method = Literal["auto", "slic", "gmm"]
METHODS: tuple[str, ...] = ("auto", "slic", "gmm")
Segmentation = SlicResult | GmmResult | AutoResult


@dataclass
class AnalysisResult:
    """Полный результат анализа снимка.

    Attributes:
        method: использованный метод сегментации.
        segmentation: ``SlicResult`` / ``GmmResult`` / ``AutoResult`` (поля ``mask``, ``labels``, ``n_rejected``).
        morphology: объекты и сводка.
        overlay: BGR-изображение с границами сегментации и эллипсами.
    """

    method: str
    segmentation: Segmentation
    morphology: MorphologyResult
    overlay: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return self.segmentation.mask

    @property
    def summary(self) -> MorphologySummary:
        return self.morphology.summary

    def status_text(self) -> str:
        return f"{self.segmentation.describe()}\n{self.summary.text()}"

    def to_dict(self, include_objects: bool = True) -> dict:
        d = {
            "method": self.method,
            "segmentation": self.segmentation.describe(),
            "bacteria_area_fraction": round(float(self.mask.mean()), 6),
            "objects_rejected_by_filter": int(self.segmentation.n_rejected),
            "summary": self.summary.to_dict(),
        }
        if isinstance(self.segmentation, SlicResult):  # параметры SLIC по тайлам
            d["slic_tiles"] = [{"bbox": list(t.bbox), "k": t.k, "m": t.m, "source": t.source}
                               for t in self.segmentation.tiles]
        if isinstance(self.segmentation, AutoResult):  # какой метод выбран в каждом тайле
            d["auto_tiles"] = [{"bbox": list(b), "method": m} for b, m in self.segmentation.tile_methods]
        if include_objects:
            d["objects"] = [o.to_dict() for o in self.morphology.objects]
        return d


class Analyzer:
    """Обёртка, хранящая загруженные модели и выполняющая анализ снимков.

    Args:
        model_path: путь к модели RF для параметров SLIC (по умолчанию ``config.MODEL_PATH``). Если файла
            нет — SLIC работает по аналитической формуле (будет предупреждение).
        predictor: можно передать уже загруженный ``SlicParamPredictor`` (тогда model_path игнорируется).
        use_filter: применять ли фильтр ложных объектов (None — по ``config.USE_OBJECT_FILTER``).
        object_filter: уже загруженный ``ObjectFilter`` (иначе грузится из ``config.OBJECT_FILTER_PATH``).
        chooser: уже загруженный ``MethodChooser`` для режима «Авто» (иначе из ``config.METHOD_CHOOSER_PATH``).
    """

    def __init__(self, model_path: str | Path = config.MODEL_PATH, predictor: SlicParamPredictor | None = None,
                 use_filter: bool | None = None, object_filter: ObjectFilter | None = None,
                 chooser: MethodChooser | None = None):
        self.predictor = predictor if predictor is not None else SlicParamPredictor.load_or_none(model_path)
        use_filter = config.USE_OBJECT_FILTER if use_filter is None else use_filter
        self.object_filter = (object_filter or ObjectFilter.load_or_none()) if use_filter else None
        self._chooser = chooser  # загружается лениво — нужен только режиму «Авто»

    @property
    def chooser(self) -> MethodChooser | None:
        if self._chooser is None:
            self._chooser = MethodChooser.load_or_none() or False
        return self._chooser or None

    def segment(self, img_bgr: np.ndarray, method: Method = "auto", **kwargs) -> Segmentation:
        """Сегментация (+ фильтр ложных объектов) + разделение на отдельные клетки (``seg.labels``).

        ``kwargs`` пробрасываются в ``segment_slic``/``segment_gmm``.
        """
        if method == "slic":
            use_analytical = kwargs.pop("use_analytical", self.predictor is None)
            seg = segment_slic(img_bgr, predictor=self.predictor, use_analytical=use_analytical, **kwargs)
        elif method == "gmm":
            seg = segment_gmm(img_bgr, **kwargs)
        elif method == "auto":
            seg = segment_auto(img_bgr, predictor=self.predictor, chooser=self.chooser)
        else:
            raise ValueError(f"Неизвестный метод сегментации: {method!r}; допустимо {METHODS}")
        if self.object_filter is not None and seg.mask.any():
            self._apply_filter(img_bgr, seg)
        # Карта ОТДЕЛЬНЫХ клеток вместо компонент связности (GMM/«Авто») и суперпикселей (SLIC):
        # её показывает GUI и её же использует морфология.
        if isinstance(seg, SlicResult):
            seg.superpixels = seg.labels
        seg.labels = split_instances(img_bgr, seg.mask)
        return seg

    def _apply_filter(self, img_bgr: np.ndarray, seg: Segmentation) -> None:
        """Удалить из маски сегментации компоненты-артефакты (на месте)."""
        kept, proba = self.object_filter.filter_mask(img_bgr, seg.mask)
        seg.n_rejected = int((proba < self.object_filter.threshold).sum())
        seg.mask = kept
        seg.labels = np.where(kept, seg.labels, 0).astype(np.int32)

    def classify(self, img_bgr: np.ndarray, segmentation: Segmentation, **kwargs) -> MorphologyResult:
        """Морфологический анализ по маске и карте клеток готовой сегментации."""
        return classify_objects(img_bgr, segmentation.mask, instances=segmentation.labels, **kwargs)

    def analyze(self, img_bgr: np.ndarray, method: Method = "auto") -> AnalysisResult:
        """Сегментация + фильтр + морфология + картинка-оверлей."""
        seg = self.segment(img_bgr, method)
        morph = self.classify(img_bgr, seg)
        overlay = render_analysis(img_bgr, seg.labels, morph)
        return AnalysisResult(method=method, segmentation=seg, morphology=morph, overlay=overlay)


def analyze_image(img_bgr: np.ndarray, method: Method = "auto", analyzer: Analyzer | None = None) -> AnalysisResult:
    """Функциональная обёртка над ``Analyzer.analyze`` (модель загружается при каждом вызове без ``analyzer``)."""
    return (analyzer or Analyzer()).analyze(img_bgr, method)
