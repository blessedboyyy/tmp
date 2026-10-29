from dataclasses import dataclass, field
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class MethodOption:
    id: str
    name: str
    segmentations: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageDefinition:
    key: str
    name: str
    methods: tuple[MethodOption, ...]
    description: str = ""


STAGES = (
    StageDefinition(
        "preprocessing", "Предобработка",
        (
            MethodOption("light_average_pool", "Усреднение 2×2"),
            MethodOption("none", "Без предобработки"),
        ),
    ),
    StageDefinition(
        "segmentation", "Сегментация",
        (
            MethodOption("light_ridges", "HSV + гребни (Light)"),
            MethodOption("imbp_otsu", "Оцу (IMBP)"),
            MethodOption("imbp_slic", "SLIC (IMBP)"),
        ),
    ),
    StageDefinition(
        "filtering", "Фильтрация",
        (
            MethodOption("light_area", "Площадь ≥ 25 (Light)", ("light_ridges",)),
            MethodOption("imbp_native", "Встроенная (IMBP)", ("imbp_otsu", "imbp_slic")),
        ),
        "Light: площадь контура ≥ 25. IMBP: встроенная очистка маски Оцу, без RF-фильтра.",
    ),
    StageDefinition(
        "instances", "Разделение объектов",
        (
            MethodOption("light_watershed", "Водораздел Light", ("light_ridges",)),
            MethodOption("connected_components", "Связные области маски", ("imbp_otsu", "imbp_slic")),
        ),
        "В IMBP пока считаются связные области маски. Соприкасающиеся клетки не разделяются.",
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


def available_methods(stage: StageDefinition, segmentation: str) -> tuple[MethodOption, ...]:
    return tuple(
        method for method in stage.methods
        if not method.segmentations or segmentation in method.segmentations
    )


def validate_number(value, minimum, maximum, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: ожидается число от {minimum} до {maximum}.")
    if not isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name}: ожидается число от {minimum} до {maximum}.")


@dataclass(frozen=True)
class OtsuSettings:
    channel: str = "flat_gray"
    threshold: float | None = None

    def __post_init__(self):
        if self.channel not in ("gray", "flat_gray", "od"):
            raise ValueError("Недоступный канал Оцу.")
        if self.threshold is not None:
            validate_number(self.threshold, 0, 255, "Порог Оцу")


@dataclass(frozen=True)
class SlicSettings:
    mode: str = "rf"
    k: int = 300
    m: float = 15.0

    def __post_init__(self):
        if self.mode not in ("manual", "analytical", "rf"):
            raise ValueError("Недоступный режим подбора параметров SLIC.")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or not 1 <= self.k <= 100000:
            raise ValueError("K должно быть целым числом от 1 до 100000.")
        validate_number(self.m, 0.01, 1000, "M")


@dataclass(frozen=True)
class PipelineSettings:
    preprocessing: str | None = None
    segmentation: str = "light_ridges"
    filtering: str | None = None
    instances: str | None = None
    morphology: str = "none"
    gram_classification: str = "none"
    sigma: float = 2.5
    otsu: OtsuSettings = field(default_factory=OtsuSettings)
    slic: SlicSettings = field(default_factory=SlicSettings)

    def __post_init__(self):
        light = self.segmentation == "light_ridges"
        defaults = {
            "preprocessing": "light_average_pool" if light else "none",
            "filtering": "light_area" if light else "imbp_native",
            "instances": "light_watershed" if light else "connected_components",
        }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)
        for stage in STAGES:
            method = getattr(self, stage.key)
            if method not in {option.id for option in available_methods(stage, self.segmentation)}:
                raise ValueError(f"Недоступный метод этапа «{stage.name}»: {method}")
        validate_number(self.sigma, 0.1, 20.0, "σ")
        if not isinstance(self.otsu, OtsuSettings) or not isinstance(self.slic, SlicSettings):
            raise ValueError("Ожидаются настройки OtsuSettings и SlicSettings.")


@dataclass(frozen=True)
class DetectedObject:
    object_id: str
    contour: np.ndarray


@dataclass(frozen=True)
class TileParameters:
    bbox: tuple[int, int, int, int]
    k: int
    m: float
    source: str


@dataclass(frozen=True)
class SegmentationResult:
    objects: tuple[DetectedObject, ...]
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    superpixels: np.ndarray | None = None
    tiles: tuple[TileParameters, ...] = ()
    count_label: str = "Объекты"
    details: str = ""


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
