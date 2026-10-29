"""Морфологическая классификация выделенных бактерий (раздел 4.3) и оценка окраски по Граму.

Вход — маска бактерий от ЛЮБОГО метода сегментации (SLIC или GMM). В исходном step12 морфология
игнорировала выбранную сегментацию и заново строила свою маску Оцу — исправлено: теперь
классифицируются именно объекты выбранной пользователем сегментации.

Шаги:
    1. Отдельные клетки: карта ``instances`` из конвейера (``Analyzer.segment``) или, если её нет,
       ``instances.split_instances`` — маркерный водораздел по рельефу «расстояние + тёмность».
       (Водораздел этапа 1 ``split_touching`` оставлен для сравнения: он склеивал до 2/3 клеток в
       плотных полях, см. reports/instances.md.)
    2. Для каждого объекта: площадь, эллипс наименьших квадратов (Fitzgibbon), эксцентриситет
       e = sqrt(1 - (b/a)^2), где a ≥ b — полуоси. ``cv2.fitEllipse`` НЕ гарантирует порядок осей,
       поэтому они сортируются (иначе под корнем отрицательное число -> NaN, как было в step11/12).
    3. Классы: площадь < ``NOISE_MIN_AREA`` — артефакт; e < ``ECCENTRICITY_THRESHOLD`` — кокк,
       иначе — бацилла (палочка).
    4. Окраска (``config.GRAM_RULE``): по умолчанию — доля оптической плотности объекта в красном
       канале > ``GRAM_OD_RED_SHARE`` -> Грам(+) (фиолетовые); правило курсовой «R > B + 15 -> Грам(−)»
       оставлено опцией "rgb" (на тест-культурах оно ошибалось на 85% снимков Грам(+)).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from . import config
from .stains import optical_density

COCCUS = "coccus"        # кокк (сферическая форма)
BACILLUS = "bacillus"    # бацилла / палочка (вытянутая форма)
ARTIFACT = "artifact"    # шум, слишком мелкий объект
GRAM_POSITIVE = "positive"
GRAM_NEGATIVE = "negative"


@dataclass
class BacteriumObject:
    """Один выделенный объект.

    Attributes:
        id: номер объекта в карте ``MorphologyResult.instances``.
        area: площадь в пикселях.
        centroid: (x, y) центр масс.
        bbox: (x, y, w, h).
        ellipse: эллипс в формате OpenCV ((cx, cy), (d1, d2), angle) или None для артефактов.
        major_axis, minor_axis: длины осей эллипса (диаметры), major ≥ minor.
        eccentricity: эксцентриситет 0..1 (0 — круг).
        morphotype: ``COCCUS`` / ``BACILLUS`` / ``ARTIFACT``.
        mean_bgr: средний цвет объекта (B, G, R).
        gram: ``GRAM_POSITIVE`` / ``GRAM_NEGATIVE`` (None для артефактов).
        od_red_share: доля оптической плотности объекта в красном канале (признак Грама).
    """

    id: int
    area: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    ellipse: tuple | None
    major_axis: float
    minor_axis: float
    eccentricity: float
    morphotype: str
    mean_bgr: tuple[float, float, float]
    gram: str | None
    od_red_share: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ellipse"] = None if self.ellipse is None else [list(self.ellipse[0]), list(self.ellipse[1]), self.ellipse[2]]
        return d


@dataclass
class MorphologySummary:
    """Сводная статистика по снимку (то, что выводится рядом с изображением в GUI)."""

    cocci: int = 0
    bacilli: int = 0
    artifacts: int = 0
    gram_positive: int = 0
    gram_negative: int = 0

    @property
    def total(self) -> int:
        """Число валидных клеток (без артефактов)."""
        return self.cocci + self.bacilli

    @property
    def gram_positive_pct(self) -> float:
        return 100.0 * self.gram_positive / self.total if self.total else 0.0

    @property
    def gram_negative_pct(self) -> float:
        return 100.0 * self.gram_negative / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "total": self.total,
                "gram_positive_pct": round(self.gram_positive_pct, 2),
                "gram_negative_pct": round(self.gram_negative_pct, 2)}

    def text(self) -> str:
        return (f"Кокки (зел.): {self.cocci} | Бациллы (син.): {self.bacilli} | Артефакты: {self.artifacts}\n"
                f"Грам(+) фиолетовые: {self.gram_positive_pct:.1f}% | Грам(−) красные: {self.gram_negative_pct:.1f}%")


@dataclass
class MorphologyResult:
    """Результат морфологического анализа.

    Attributes:
        instances: int32 (H, W) — карта отдельных объектов после водораздела (0 — фон).
        objects: список объектов (включая артефакты).
        summary: сводная статистика.
    """

    instances: np.ndarray
    objects: list[BacteriumObject] = field(default_factory=list)
    summary: MorphologySummary = field(default_factory=MorphologySummary)


# ---------------------------------------------------------------------------- шаги

def split_touching(mask: np.ndarray, fg_fraction: float = config.WATERSHED_FG_FRACTION,
                   min_area: int = config.WATERSHED_MIN_AREA) -> np.ndarray:
    """Разделить слипшиеся объекты маски водоразделом. Возвращает карту экземпляров int32."""
    mask = mask.astype(bool)
    comp, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return comp.astype(np.int32)

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    idx = np.arange(1, n + 1)
    comp_max = np.zeros(n + 1)
    comp_max[1:] = ndi.maximum(dist, comp, idx)
    comp_area = np.bincount(comp.ravel(), minlength=n + 1)

    # «Ядра» клеток: внутри каждой компоненты — относительно её собственного максимума расстояния.
    sure_fg = mask & (dist > fg_fraction * comp_max[comp])
    # Мелкие компоненты не режем: вся компонента — один маркер.
    small = comp_area[comp] < min_area
    sure_fg |= mask & small
    markers, _ = ndi.label(sure_fg, structure=np.ones((3, 3)))

    return watershed(-dist, markers, mask=mask).astype(np.int32)


def ellipse_eccentricity(contour: np.ndarray) -> tuple[tuple, float, float, float] | None:
    """Эллипс по контуру и его эксцентриситет: (ellipse, major, minor, e) или None, если точек < 5."""
    if len(contour) < 5:
        return None
    ellipse = cv2.fitEllipse(contour)
    d1, d2 = ellipse[1]
    major, minor = max(d1, d2), min(d1, d2)
    if major <= 0:
        return None
    e = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))
    return ellipse, float(major), float(minor), e


def gram_from_color(mean_bgr: tuple[float, float, float], red_delta: float = config.GRAM_RED_DELTA) -> str:
    """Правило курсовой: красный оттенок (R > B + delta) -> Грам(−), иначе Грам(+)."""
    b, _, r = mean_bgr
    return GRAM_NEGATIVE if r > b + red_delta else GRAM_POSITIVE


def gram_from_od(mean_od_bgr: tuple[float, float, float], red_share: float = config.GRAM_OD_RED_SHARE) -> str:
    """Грам по оптической плотности: доля OD в красном канале > порога -> Грам(+), иначе Грам(−)."""
    total = sum(mean_od_bgr)
    share = mean_od_bgr[2] / total if total > 0 else 0.0
    return GRAM_POSITIVE if share > red_share else GRAM_NEGATIVE


def classify_objects(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    split: bool = True,
    instances: np.ndarray | None = None,
    noise_min_area: int = config.NOISE_MIN_AREA,
    ecc_threshold: float = config.ECCENTRICITY_THRESHOLD,
    red_delta: float = config.GRAM_RED_DELTA,
    gram_rule: str | None = None,
) -> MorphologyResult:
    """Найти отдельные бактерии в маске и классифицировать их по форме и окраске.

    Args:
        img_bgr: исходный снимок BGR uint8 (нужен для цвета).
        mask: bool (H, W) — маска бактерий от ``segment_slic``/``segment_gmm``.
        split: разделять ли слипшиеся клетки (если ``instances`` не переданы).
        instances: готовая карта отдельных клеток int (0 — фон), например ``seg.labels`` после
            ``Analyzer.segment``. Если передана — повторно не вычисляется.
        noise_min_area, ecc_threshold, red_delta: пороги (по умолчанию из ``config``).
        gram_rule: "od_red" или "rgb" (None — ``config.GRAM_RULE``).
    """
    gram_rule = config.GRAM_RULE if gram_rule is None else gram_rule
    # Оптическая плотность считается один раз на снимок (фон оценивается по всему снимку).
    od = optical_density(img_bgr) if gram_rule == "od_red" and mask.any() else None
    if instances is None:
        from .instances import split_instances  # локальный импорт: instances -> masks -> stains

        instances = split_instances(img_bgr, mask) if split else ndi.label(mask, structure=np.ones((3, 3)))[0]
    instances = np.where(mask, instances, 0).astype(np.int32)
    result = MorphologyResult(instances=instances)
    summary = result.summary

    for obj_id, slc in enumerate(ndi.find_objects(instances), start=1):
        if slc is None:
            continue
        local = instances[slc] == obj_id
        area = int(local.sum())
        ys, xs = np.nonzero(local)
        y_off, x_off = slc[0].start, slc[1].start
        centroid = (float(xs.mean() + x_off), float(ys.mean() + y_off))
        bbox = (x_off, y_off, local.shape[1], local.shape[0])
        pixels = img_bgr[slc][local]
        mean_bgr = tuple(float(v) for v in pixels.mean(axis=0))
        mean_od = tuple(float(v) for v in od[slc][local].mean(axis=0)) if od is not None else (0.0, 0.0, 0.0)
        od_red_share = mean_od[2] / sum(mean_od) if sum(mean_od) > 0 else 0.0

        fitted = None
        if area >= noise_min_area:
            contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                cnt = max(contours, key=cv2.contourArea) + np.array([x_off, y_off])
                fitted = ellipse_eccentricity(cnt.astype(np.int32))

        if fitted is None:
            summary.artifacts += 1
            result.objects.append(BacteriumObject(obj_id, area, centroid, bbox, None, 0.0, 0.0, 0.0,
                                                  ARTIFACT, mean_bgr, None, od_red_share))
            continue

        ellipse, major, minor, ecc = fitted
        morphotype = COCCUS if ecc < ecc_threshold else BACILLUS
        gram = gram_from_od(mean_od) if gram_rule == "od_red" else gram_from_color(mean_bgr, red_delta)
        if morphotype == COCCUS:
            summary.cocci += 1
        else:
            summary.bacilli += 1
        if gram == GRAM_NEGATIVE:
            summary.gram_negative += 1
        else:
            summary.gram_positive += 1
        result.objects.append(BacteriumObject(obj_id, area, centroid, bbox, ellipse, major, minor, ecc,
                                              morphotype, mean_bgr, gram, od_red_share))
    return result
