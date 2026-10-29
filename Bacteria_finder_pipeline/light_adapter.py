import numpy as np

from Bacteria_finder_core_light.BFmainlib import BF_image

from .models import DetectedObject, PipelineSettings, SegmentationResult


class LightAdapter:
    def __init__(self):
        self._segmentor = BF_image(verbose=False)

    def prepare(self, image_bgr: np.ndarray, preprocessing: str) -> np.ndarray:
        self._segmentor.load_image(how="image", image=image_bgr)
        if preprocessing == "light_average_pool":
            self._segmentor.preprocess_image()
        else:
            self._segmentor.bacteria_image_preprocessed = self._segmentor.bacteria_image_loaded.copy()
        return np.ascontiguousarray(
            self._segmentor.bacteria_image_preprocessed[:, :, ::-1]
        )

    def segment(self, settings: PipelineSettings) -> SegmentationResult:
        self._segmentor.segment_image(type="ridges", sigma=settings.sigma)
        objects = tuple(
            DetectedObject(obj.object_id, obj.object_countour_coords.copy())
            for obj in self._segmentor.objects_db.values()
        )
        return SegmentationResult(objects=objects)
