from collections.abc import Callable
from time import perf_counter

import cv2
import numpy as np

from .light_adapter import LightAdapter
from .imbp_adapter import ImbpAdapter
from .models import PipelineSettings, ProcessingResult


ADAPTERS = {"light_ridges": LightAdapter, "imbp_otsu": ImbpAdapter, "imbp_slic": ImbpAdapter}


class ProcessingCancelled(Exception):
    pass


def run_pipeline(
    image_bgr: np.ndarray,
    settings: PipelineSettings | None = None,
    on_progress: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ProcessingResult:
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError("Ожидается изображение BGR в виде массива NumPy.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError("Ожидается непустое трёхканальное изображение BGR.")
    if image_bgr.dtype != np.uint8:
        raise ValueError("Изображение BGR должно иметь тип uint8.")
    settings = settings if settings is not None else PipelineSettings()
    if not isinstance(settings, PipelineSettings):
        raise ValueError("Ожидаются настройки PipelineSettings.")

    def checkpoint(message: str = ""):
        if should_stop is not None and should_stop():
            raise ProcessingCancelled()
        if message and on_progress is not None:
            on_progress(message)

    started = perf_counter()
    checkpoint("Подготовка изображения…")
    adapter = ADAPTERS[settings.segmentation]()
    processed_image_bgr = adapter.prepare(image_bgr, settings.preprocessing)
    checkpoint("Сегментация…")
    segmentation = adapter.segment(settings)
    checkpoint("Отрисовка результата…")
    contours = [obj.contour for obj in segmentation.objects]
    if segmentation.mask is not None:
        contours, _ = cv2.findContours(
            segmentation.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
    overlay = cv2.drawContours(
        image_bgr.copy(),
        contours,
        -1, (0, 255, 0), 1,
    )
    checkpoint()
    return ProcessingResult(
        settings=settings,
        source_shape=image_bgr.shape[:2],
        processed_image_bgr=processed_image_bgr,
        segmentation=segmentation,
        overlay_image_bgr=overlay,
        elapsed_seconds=perf_counter() - started,
    )
