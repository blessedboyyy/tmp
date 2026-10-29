from collections.abc import Callable
from time import perf_counter

import cv2
import numpy as np

from .light_adapter import LightAdapter
from .models import PipelineSettings, ProcessingResult


ADAPTERS = {"light_ridges": LightAdapter}


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
    checkpoint("Предобработка Light…")
    adapter = ADAPTERS[settings.segmentation]()
    processed_image_bgr = adapter.prepare(image_bgr)
    checkpoint("Сегментация Light…")
    segmentation = adapter.segment(settings.sigma)
    checkpoint("Отрисовка результата…")
    overlay = cv2.drawContours(
        image_bgr.copy(),
        [obj.contour for obj in segmentation.objects],
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
