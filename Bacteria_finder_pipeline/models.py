from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class MethodOption:
    id: str
    name: str


@dataclass(frozen=True)
class StageDefinition:
    key: str
    name: str
    methods: tuple[MethodOption, ...]
    description: str = ""


STAGES = (
    StageDefinition(
        "preprocessing", "Предобработка",
        (MethodOption("light_average_pool", "Усреднение 2×2"),),
    ),
    StageDefinition(
        "segmentation", "Сегментация",
        (MethodOption("light_ridges", "HSV + гребни (Light)"),),
    ),
    StageDefinition(
        "filtering", "Фильтрация",
        (MethodOption("light_area", "Площадь ≥ 25 (Light)"),),
        "Фильтрация по площади контура входит в сегментацию Light.",
    ),
    StageDefinition(
        "instances", "Разделение объектов",
        (MethodOption("light_watershed", "Водораздел Light"),),
        "Разделение объектов входит в сегментацию Light.",
    ),
    StageDefinition(
        "morphology", "Морфология",
        (MethodOption("none", "Не выполнять"),),
    ),
    StageDefinition(
        "gram_classification", "Классификация по Граму",
        (MethodOption("none", "Не выполнять"),),
    ),
)


@dataclass(frozen=True)
class PipelineSettings:
    preprocessing: str = "light_average_pool"
    segmentation: str = "light_ridges"
    filtering: str = "light_area"
    instances: str = "light_watershed"
    morphology: str = "none"
    gram_classification: str = "none"
    sigma: float = 2.5

    def __post_init__(self):
        for stage in STAGES:
            method = getattr(self, stage.key)
            if method not in {option.id for option in stage.methods}:
                raise ValueError(f"Недоступный метод этапа «{stage.name}»: {method}")
        if not isinstance(self.sigma, (int, float)) or isinstance(self.sigma, bool):
            raise ValueError("σ должна быть числом от 0.1 до 20.0.")
        if not isfinite(self.sigma) or not 0.1 <= self.sigma <= 20.0:
            raise ValueError("σ должна быть числом от 0.1 до 20.0.")


@dataclass(frozen=True)
class DetectedObject:
    object_id: str
    contour: np.ndarray


@dataclass(frozen=True)
class SegmentationResult:
    objects: tuple[DetectedObject, ...]
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None


@dataclass(frozen=True)
class ProcessingResult:
    settings: PipelineSettings
    source_shape: tuple[int, int]
    processed_image_bgr: np.ndarray
    segmentation: SegmentationResult
    overlay_image_bgr: np.ndarray
    elapsed_seconds: float

    @property
    def object_count(self) -> int:
        return len(self.segmentation.objects)
