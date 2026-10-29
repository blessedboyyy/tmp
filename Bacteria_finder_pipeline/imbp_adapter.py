from functools import lru_cache
from pathlib import Path
import warnings

import cv2
import numpy as np

from .light_adapter import LightAdapter
from .models import DetectedObject, PipelineSettings, SegmentationResult, TileParameters


def load_imbp():
    try:
        from third_party.imbp.src import imbp
        return imbp
    except ImportError as exc:
        raise RuntimeError(
            "Не удалось загрузить IMBP. Установите зависимости из requirements.txt "
            "в интерпретатор, которым запускается программа."
        ) from exc


@lru_cache(maxsize=2)
def _load_predictor(path: str, modified_ns: int, size: int):
    imbp = load_imbp()
    import sklearn
    from sklearn.exceptions import InconsistentVersionWarning

    if sklearn.__version__ != "1.9.1":
        raise RuntimeError(
            "Для комплектной RF-модели требуется scikit-learn==1.9.1. "
            "Установите зависимости из requirements.txt."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            predictor = imbp.SlicParamPredictor.load(path)
        if not predictor.is_trained or predictor.meta.get("sklearn_version") != "1.9.1":
            raise ValueError("Отсутствует обученная модель либо не совпадает версия обучения.")
        if predictor.meta.get("inputs") != list(imbp.params_model.MODEL_INPUTS):
            raise ValueError("Не совпадают имена или порядок входных признаков.")
        if getattr(predictor.model, "n_features_in_", None) != 8:
            raise ValueError("Ожидалось восемь входных признаков RF.")
        if not np.array_equal(predictor.grid, imbp.params_model.param_grid()):
            raise ValueError("Не совпадает сетка параметров SLIC.")
        return predictor
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось загрузить RF-модель SLIC: {path}. "
            "Восстановите комплектный файл slic_params_rf.joblib. "
            "Можно отдельно выбрать ручной или аналитический режим."
        ) from exc


def load_slic_predictor():
    imbp = load_imbp()
    path = Path(imbp.config.MODEL_PATH).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise RuntimeError(
            f"RF-модель SLIC недоступна: {path}. "
            "Восстановите файл или выберите ручной/аналитический режим."
        ) from exc
    return _load_predictor(str(path), stat.st_mtime_ns, stat.st_size)


def mask_result(mask, *, superpixels=None, tiles=(), details=""):
    mask = np.ascontiguousarray(mask, dtype=bool).copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    objects = []
    for object_id in range(1, count):
        x, y, width, height = stats[object_id, :4]
        local = (labels[y:y + height, x:x + width] == object_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(int(x), int(y))
        )
        objects.append(DetectedObject(str(object_id), max(contours, key=cv2.contourArea)))
    return SegmentationResult(
        objects=tuple(objects), mask=mask, labels=labels,
        superpixels=None if superpixels is None else superpixels.copy(),
        tiles=tiles, count_label="Компоненты маски", details=details,
    )


class ImbpAdapter:
    def prepare(self, image_bgr: np.ndarray, preprocessing: str) -> np.ndarray:
        if min(image_bgr.shape[:2]) < 4:
            raise ValueError("Для IMBP ширина и высота изображения должны быть не меньше 4 пикселей.")
        if preprocessing == "light_average_pool":
            self._image = LightAdapter().prepare(image_bgr, preprocessing)
        else:
            self._image = np.ascontiguousarray(image_bgr).copy()
        return self._image

    def segment(self, settings: PipelineSettings) -> SegmentationResult:
        imbp = load_imbp()
        if settings.segmentation == "imbp_otsu":
            native = imbp.masks.otsu_bacteria_mask(
                self._image, channel=settings.otsu.channel, threshold=settings.otsu.threshold
            )
            mode = "авто" if settings.otsu.threshold is None else "ручной"
            return mask_result(
                native.mask,
                details=f"Оцу: порог {native.threshold:g} ({mode}); разница классов {native.class_gap:.2f}.",
            )
        mode = settings.slic.mode
        native = imbp.segment_slic(
            self._image,
            predictor=load_slic_predictor() if mode == "rf" else None,
            fixed_params=(settings.slic.k, settings.slic.m) if mode == "manual" else None,
            use_analytical=mode == "analytical",
        )
        tiles = tuple(TileParameters(t.bbox, t.k, t.m, t.source) for t in native.tiles)
        names = {"rf": "RF", "analytical": "аналитика", "manual": "ручной"}
        details = f"SLIC ({names[mode]}): обработано тайлов {len(tiles)}."
        if tiles:
            ks = ", ".join(str(k) for k in sorted({t.k for t in tiles}))
            ms = ", ".join(f"{m:.3f}".rstrip("0").rstrip(".") for m in sorted({t.m for t in tiles}))
            details += f" K: {ks}; M: {ms}."
        else:
            details += " Все тайлы пропущены по маске Оцу."
        return mask_result(native.mask, superpixels=native.labels, tiles=tiles, details=details)
