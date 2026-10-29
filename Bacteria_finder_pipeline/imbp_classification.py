from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from .imbp_adapter import load_imbp
from .models import ClassificationResult, ClassifiedObject


def light_label_groups(shape, objects):
    groups = []
    for index, obj in enumerate(objects, start=1):
        x, y, w, h = cv2.boundingRect(obj.contour)
        local = np.zeros((h, w), np.uint8)
        cv2.drawContours(local, [obj.contour], -1, 1, -1, offset=(-x, -y))
        selected = local.astype(bool)
        for labels in groups:
            if not np.any(labels[y:y + h, x:x + w][selected]):
                break
        else:
            labels = np.zeros(shape, np.int32)
            groups.append(labels)
        labels[y:y + h, x:x + w][selected] = index
    return groups


def classify_segmentation(image_bgr, segmentation, settings, checkpoint=lambda: None):
    if settings.morphology == "none" and settings.gram_classification == "none":
        return None
    imbp = load_imbp()
    params = settings.classification_params
    if segmentation.labels is None:
        groups = light_label_groups(image_bgr.shape[:2], segmentation.objects)
        ids = {i: obj.object_id for i, obj in enumerate(segmentation.objects, start=1)}
    else:
        groups = [segmentation.labels]
        ids = {int(obj.object_id): obj.object_id for obj in segmentation.objects}
    objects = []
    gram_rule = "rgb" if settings.gram_classification == "imbp_rgb" else "od_red"
    for labels in groups:
        checkpoint()
        mask = labels > 0 if segmentation.mask is None else segmentation.mask
        native = imbp.classify_objects(
            image_bgr, mask, instances=labels, split=False,
            noise_min_area=params.noise_min_area, ecc_threshold=params.ecc_threshold,
            red_delta=params.red_delta, gram_rule=gram_rule,
        )
        for obj in native.objects:
            objects.append(ClassifiedObject(
                object_id=ids[obj.id], area=obj.area, centroid=obj.centroid, bbox=obj.bbox,
                ellipse=obj.ellipse, major_axis=obj.major_axis, minor_axis=obj.minor_axis,
                eccentricity=obj.eccentricity, mean_bgr=obj.mean_bgr,
                artifact=obj.morphotype == imbp.morphology.ARTIFACT,
                morphotype=obj.morphotype if settings.morphology != "none" else None,
                gram=obj.gram if settings.gram_classification != "none" else None,
                od_red_share=obj.od_red_share if settings.gram_classification == "imbp_od" else None,
            ))
    order = {obj.object_id: i for i, obj in enumerate(segmentation.objects)}
    objects.sort(key=lambda obj: order[obj.object_id])
    checkpoint()
    return ClassificationResult(tuple(objects), settings.morphology, settings.gram_classification)


def render_classification(image_bgr, segmentation, classification, segmentation_overlay):
    if classification is None:
        return None, None
    imbp = load_imbp()
    boundaries = (
        imbp.visualization.draw_boundaries(image_bgr, segmentation.labels)
        if segmentation.labels is not None else segmentation_overlay
    )
    morphology_image = None
    if classification.morphology != "none":
        morphology_image = imbp.visualization.draw_objects(boundaries, classification)
    gram_image = None
    if classification.gram_classification != "none":
        gram_image = boundaries.copy()
        for obj in classification.objects:
            if obj.artifact:
                continue
            color = (200, 0, 200) if obj.gram == "positive" else (0, 0, 255)
            cv2.ellipse(gram_image, obj.ellipse, color, 2)
    return morphology_image, gram_image


def classification_text(result):
    classification = result.classification
    if classification is None:
        return ""
    s = classification.summary
    lines = [f"Объектов: {s['objects']}; без артефактов: {s['valid']}; артефактов: {s['artifacts']}."]
    if classification.morphology != "none":
        lines.append(f"Кокки (зелёный): {s['cocci']}; палочки (синий): {s['bacilli']}.")
    if classification.gram_classification != "none":
        lines.append(
            f"Грам+ (фиолетовый): {s['gram_positive']} ({s['gram_positive_pct']:.1f}%); "
            f"Грам− (красный): {s['gram_negative']} ({s['gram_negative_pct']:.1f}%)."
        )
        if not s['valid']:
            lines.append("Нет объектов, пригодных для оценки окраски.")
    if result.settings.instances == "connected_components":
        lines.append("Анализ связных областей. Для разделения соприкасающихся клеток выберите водораздел.")
    return "\n".join(lines)


def export_classification(result, path):
    if result.classification is None:
        raise ValueError("Сначала выполните классификацию.")
    content = {
        "settings": asdict(result.settings),
        "image_shape": result.source_shape,
        "segmentation_objects": result.object_count,
        "objects_rejected_by_rf": result.segmentation.n_rejected,
        "rf_threshold": result.segmentation.filter_threshold,
        "segmentation_details": result.segmentation.details,
        "summary": result.classification.summary,
        "objects": [asdict(obj) for obj in result.classification.objects],
    }
    text = json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")
