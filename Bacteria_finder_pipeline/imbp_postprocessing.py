from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import warnings

import cv2
import numpy as np
from scipy import ndimage as ndi

from .imbp_adapter import load_imbp, mask_result
from .models import DetectedObject, InstanceSettings, ObjectFilterSettings, validate_number


@lru_cache(maxsize=2)
def _load_object_filter(path: str, modified_ns: int, size: int):
    imbp = load_imbp()
    import sklearn
    from sklearn.exceptions import InconsistentVersionWarning

    if sklearn.__version__ != "1.9.1":
        raise RuntimeError(
            "Для комплектного RF-фильтра требуется scikit-learn==1.9.1. "
            "Установите зависимости из requirements.txt."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = imbp.object_filter.ObjectFilter.load(path)
        if model.model is None or model.meta.get("sklearn_version") != "1.9.1":
            raise ValueError("Отсутствует обученный фильтр либо не совпадает версия обучения.")
        if model.meta.get("features") != list(imbp.object_filter.OBJECT_FEATURES):
            raise ValueError("Не совпадают имена или порядок признаков RF-фильтра.")
        if getattr(model.model, "n_features_in_", None) != len(imbp.object_filter.OBJECT_FEATURES):
            raise ValueError("Не совпадает число входных признаков RF-фильтра.")
        if not np.array_equal(getattr(model.model, "classes_", None), [0, 1]):
            raise ValueError("Не совпадают классы RF-фильтра.")
        validate_number(model.threshold, 0, 1, "Порог RF-модели")
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось загрузить RF-фильтр: {path}. "
            "Восстановите object_filter_rf.joblib или выберите встроенную фильтрацию без RF."
        ) from exc


def load_object_filter():
    imbp = load_imbp()
    path = Path(imbp.config.OBJECT_FILTER_PATH).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise RuntimeError(
            f"RF-фильтр недоступен: {path}. "
            "Восстановите object_filter_rf.joblib или выберите встроенную фильтрацию без RF."
        ) from exc
    return _load_object_filter(str(path), stat.st_mtime_ns, stat.st_size)


def filter_objects(image_bgr, segmentation, settings: ObjectFilterSettings):
    model = load_object_filter()
    if settings.threshold is not None:
        model = replace(model, threshold=settings.threshold)
    if segmentation.mask.any():
        kept, probabilities = model.filter_mask(image_bgr, segmentation.mask)
    else:
        kept = segmentation.mask.copy()
        probabilities = np.zeros(0)
    mapped = mask_result(kept)
    rejected = int((probabilities < model.threshold).sum())
    superpixels = segmentation.superpixels
    if superpixels is not None:
        superpixels = np.where(kept, superpixels, 0).astype(superpixels.dtype)
    details = f"RF: удалено {rejected} из {len(probabilities)} областей; порог {model.threshold:g}."
    return replace(
        segmentation, objects=mapped.objects, mask=mapped.mask, labels=mapped.labels,
        superpixels=superpixels, filter_probabilities=probabilities.copy(),
        filter_threshold=model.threshold, n_rejected=rejected,
        details=f"{segmentation.details}\n{details}".strip(),
    )


def objects_from_labels(labels):
    objects = []
    for object_id, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        local = (labels[slices] == object_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            offset=(int(xs.start), int(ys.start)),
        )
        objects.append(DetectedObject(str(object_id), max(contours, key=cv2.contourArea)))
    return tuple(objects)


def split_objects(image_bgr, segmentation, settings: InstanceSettings):
    if segmentation.mask.all():
        raise ValueError(
            "Водораздел IMBP не поддерживает маску, занимающую всё изображение без фона. "
            "Измените параметры сегментации или выберите «Связные области маски»."
        )
    imbp = load_imbp()
    labels = imbp.instances.split_instances(
        image_bgr, segmentation.mask, h=settings.h, sigma=settings.sigma,
        beta=settings.beta, neck_ratio=settings.neck_ratio,
    )
    if not np.array_equal(labels > 0, segmentation.mask):
        raise ValueError(
            "Водораздел IMBP вернул метки, не совпадающие с маской. "
            "Выберите «Связные области маски» и проверьте параметры сегментации."
        )
    details = (
        f"Водораздел IMBP: h={settings.h:g}, σ={settings.sigma:g}, "
        f"β={settings.beta:g}, склейка={settings.neck_ratio:g}."
    )
    return replace(
        segmentation, objects=objects_from_labels(labels), labels=labels.copy(),
        count_label="Клетки", details=f"{segmentation.details}\n{details}".strip(),
    )
