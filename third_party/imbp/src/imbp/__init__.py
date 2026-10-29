"""imbp — сегментация и классификация микробиоты на микроснимках с окраской по Граму.

Реализация алгоритма курсовой «Применение методов компьютерного зрения для сегментации и
классификации микробиоты в космических исследованиях» (МГУ, ФКИ, 2026).

Публичное API (всё остальное — внутренние детали):
    Analyzer, analyze_image, AnalysisResult  — полный конвейер (pipeline.py)
    segment_slic, segment_gmm                — методы сегментации (segmentation/)
    classify_objects                         — морфология и Грам (morphology.py)
    SlicParamPredictor                       — модель RF для параметров SLIC (params_model.py)
    extract_features                         — признаки изображения (features.py)
    load_image, save_image                   — ввод-вывод с поддержкой кириллицы (io.py)
    render_analysis                          — отрисовка результата (visualization.py)
"""

from .features import extract_features
from .io import load_image, load_mask, save_image
from .morphology import BacteriumObject, MorphologyResult, MorphologySummary, classify_objects
from .params_model import SlicParamPredictor
from .pipeline import METHODS, AnalysisResult, Analyzer, analyze_image
from .segmentation import GmmResult, SlicResult, segment_gmm, segment_slic
from .visualization import render_analysis

__version__ = "1.0.0"

__all__ = [
    "METHODS",
    "AnalysisResult",
    "Analyzer",
    "BacteriumObject",
    "GmmResult",
    "MorphologyResult",
    "MorphologySummary",
    "SlicParamPredictor",
    "SlicResult",
    "analyze_image",
    "classify_objects",
    "extract_features",
    "load_image",
    "load_mask",
    "render_analysis",
    "save_image",
    "segment_gmm",
    "segment_slic",
]
