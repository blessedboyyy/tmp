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
            MethodOption("imbp_gmm", "GMM (IMBP)"),
            MethodOption("imbp_auto", "Авто (IMBP)"),
        ),
    ),
    StageDefinition(
        "filtering", "Фильтрация",
        (
            MethodOption("light_area", "Площадь ≥ 25 (Light)", ("light_ridges",)),
            MethodOption("imbp_native", "Встроенная (IMBP)", ("imbp_otsu", "imbp_slic", "imbp_gmm", "imbp_auto")),
            MethodOption("imbp_rf", "Встроенная + RF", ("imbp_otsu", "imbp_slic", "imbp_gmm", "imbp_auto")),
        ),
        "IMBP: встроенная очистка сохраняется; RF дополнительно удаляет ложные объекты после сегментации.",
    ),
    StageDefinition(
        "instances", "Разделение объектов",
        (
            MethodOption("light_watershed", "Водораздел Light", ("light_ridges",)),
            MethodOption("connected_components", "Связные области маски", ("imbp_otsu", "imbp_slic", "imbp_gmm", "imbp_auto")),
            MethodOption("imbp_watershed", "Водораздел (IMBP)", ("imbp_otsu", "imbp_slic", "imbp_gmm", "imbp_auto")),
        ),
        "Связные области сохраняют слипшиеся клетки; водораздел IMBP выделяет отдельные клетки после фильтрации.",
    ),
    StageDefinition(
        "morphology", "Морфология",
        (MethodOption("none", "Не выполнять"), MethodOption("imbp_ellipse", "Эллипсы (IMBP)")),
        "Кокки и палочки по эллипсу объекта; артефакты учитываются отдельно.",
    ),
    StageDefinition(
        "gram_classification", "Классификация по Граму",
        (
            MethodOption("none", "Не выполнять"),
            MethodOption("imbp_od", "OD (IMBP)"),
            MethodOption("imbp_rgb", "RGB (IMBP)"),
        ),
        "Окраска по оптической плотности или правилу R > B + Δ; артефакты исключены из процентов.",
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
class GmmSettings:
    n_components: int = 3
    min_area: int = 50
    bilateral: bool = True
    hmrf_beta: float = 0.0

    def __post_init__(self):
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, int) or not 2 <= self.n_components <= 8:
            raise ValueError("Число кластеров GMM должно быть целым числом от 2 до 8.")
        if isinstance(self.min_area, bool) or not isinstance(self.min_area, int) or not 0 <= self.min_area <= 100000:
            raise ValueError("Минимальная площадь GMM должна быть целым числом от 0 до 100000.")
        if not isinstance(self.bilateral, bool):
            raise ValueError("Параметр bilateral должен быть логическим значением.")
        validate_number(self.hmrf_beta, 0, 50, "β HMRF")


@dataclass(frozen=True)
class ObjectFilterSettings:
    threshold: float | None = None

    def __post_init__(self):
        if self.threshold is not None:
            validate_number(self.threshold, 0, 1, "Порог RF")


@dataclass(frozen=True)
class InstanceSettings:
    h: float = 1.5
    sigma: float = 0.5
    beta: float = 0.2
    neck_ratio: float = 1.0

    def __post_init__(self):
        validate_number(self.h, 0.01, 100, "h водораздела")
        validate_number(self.sigma, 0, 20, "σ водораздела")
        validate_number(self.beta, 0, 5, "β тёмности")
        validate_number(self.neck_ratio, 0, 10, "Порог склейки")


@dataclass(frozen=True)
class ClassificationSettings:
    noise_min_area: int = 15
    ecc_threshold: float = 0.7
    red_delta: float = 15.0

    def __post_init__(self):
        if isinstance(self.noise_min_area, bool) or not isinstance(self.noise_min_area, int) or not 0 <= self.noise_min_area <= 100000:
            raise ValueError("Минимальная площадь классификации должна быть целым числом от 0 до 100000.")
        validate_number(self.ecc_threshold, 0, 1, "Порог эксцентриситета")
        validate_number(self.red_delta, 0, 255, "Δ красного канала")


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
    gmm: GmmSettings = field(default_factory=GmmSettings)
    object_filter: ObjectFilterSettings = field(default_factory=ObjectFilterSettings)
    instance_params: InstanceSettings = field(default_factory=InstanceSettings)
    classification_params: ClassificationSettings = field(default_factory=ClassificationSettings)

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
        if not isinstance(self.gmm, GmmSettings):
            raise ValueError("Ожидаются настройки GmmSettings.")
        if not isinstance(self.object_filter, ObjectFilterSettings) or not isinstance(self.instance_params, InstanceSettings):
            raise ValueError("Ожидаются настройки ObjectFilterSettings и InstanceSettings.")
        if not isinstance(self.classification_params, ClassificationSettings):
            raise ValueError("Ожидаются настройки ClassificationSettings.")


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
class TileMethod:
    bbox: tuple[int, int, int, int]
    method: str


@dataclass(frozen=True)
class SegmentationResult:
    objects: tuple[DetectedObject, ...]
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    superpixels: np.ndarray | None = None
    tiles: tuple[TileParameters, ...] = ()
    count_label: str = "Объекты"
    details: str = ""
    clusters: np.ndarray | None = None
    cluster_means: np.ndarray | None = None
    bacteria_clusters: tuple[int, ...] = ()
    tile_methods: tuple[TileMethod, ...] = ()
    filter_probabilities: np.ndarray | None = None
    filter_threshold: float | None = None
    n_rejected: int = 0


@dataclass(frozen=True)
class ClassifiedObject:
    object_id: str
    area: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    ellipse: tuple | None
    major_axis: float
    minor_axis: float
    eccentricity: float
    mean_bgr: tuple[float, float, float]
    artifact: bool
    morphotype: str | None
    gram: str | None
    od_red_share: float | None


@dataclass(frozen=True)
class ClassificationResult:
    objects: tuple[ClassifiedObject, ...]
    morphology: str
    gram_classification: str

    @property
    def summary(self) -> dict:
        artifacts = sum(obj.artifact for obj in self.objects)
        valid = len(self.objects) - artifacts
        summary = {"objects": len(self.objects), "valid": valid, "artifacts": artifacts}
        if self.morphology != "none":
            summary.update(
                cocci=sum(obj.morphotype == "coccus" for obj in self.objects),
                bacilli=sum(obj.morphotype == "bacillus" for obj in self.objects),
            )
        if self.gram_classification != "none":
            positive = sum(obj.gram == "positive" for obj in self.objects)
            negative = sum(obj.gram == "negative" for obj in self.objects)
            summary.update(
                gram_positive=positive, gram_negative=negative,
                gram_positive_pct=100.0 * positive / valid if valid else 0.0,
                gram_negative_pct=100.0 * negative / valid if valid else 0.0,
            )
        return summary


@dataclass(frozen=True)
class ProcessingResult:
    settings: PipelineSettings
    source_shape: tuple[int, int]
    processed_image_bgr: np.ndarray
    segmentation: SegmentationResult
    overlay_image_bgr: np.ndarray
    elapsed_seconds: float
    classification: ClassificationResult | None = None
    morphology_image_bgr: np.ndarray | None = None
    gram_image_bgr: np.ndarray | None = None

    @property
    def object_count(self) -> int:
        return len(self.segmentation.objects)
