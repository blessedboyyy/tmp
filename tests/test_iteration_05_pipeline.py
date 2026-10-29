from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from Bacteria_finder_pipeline import (
    ClassificationSettings, DetectedObject, PipelineSettings, ProcessingCancelled,
    SegmentationResult, run_pipeline,
)
from Bacteria_finder_pipeline import imbp_classification as classification
from Bacteria_finder_pipeline.imbp_postprocessing import objects_from_labels
from third_party.imbp.src import imbp


ROOT = Path(__file__).resolve().parents[1]


class ClassificationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imread(str(ROOT / "test_input_1.bmp"))
        cls.crop = cls.image[700:981, 500:789].copy()

    def settings(self, **kwargs):
        return replace(PipelineSettings(
            segmentation="imbp_otsu", instances="imbp_watershed",
            morphology="imbp_ellipse", gram_classification="imbp_od",
        ), **kwargs)

    def assert_native(self, got, native, morph=True, gram="imbp_od"):
        self.assertEqual(len(got.objects), len(native.objects))
        by_id = {obj.object_id: obj for obj in got.objects}
        for expected in native.objects:
            obj = by_id[str(expected.id)]
            self.assertEqual(obj.area, expected.area)
            self.assertEqual(obj.bbox, expected.bbox)
            self.assertEqual(obj.ellipse, expected.ellipse)
            np.testing.assert_array_equal(obj.mean_bgr, expected.mean_bgr)
            np.testing.assert_array_equal(obj.centroid, expected.centroid)
            self.assertEqual(obj.eccentricity, expected.eccentricity)
            self.assertEqual(obj.major_axis, expected.major_axis)
            self.assertEqual(obj.minor_axis, expected.minor_axis)
            self.assertEqual(obj.artifact, expected.morphotype == "artifact")
            self.assertEqual(obj.morphotype, expected.morphotype if morph else None)
            self.assertEqual(obj.gram, expected.gram if gram != "none" else None)
            self.assertEqual(obj.od_red_share, expected.od_red_share if gram == "imbp_od" else None)
        s = got.summary
        self.assertEqual(s["valid"], native.summary.total)
        self.assertEqual(s["artifacts"], native.summary.artifacts)
        if morph:
            self.assertEqual(s["cocci"], native.summary.cocci)
            self.assertEqual(s["bacilli"], native.summary.bacilli)
        else:
            self.assertNotIn("cocci", s)
        if gram != "none":
            self.assertEqual(s["gram_positive"], native.summary.gram_positive)
            self.assertEqual(s["gram_negative"], native.summary.gram_negative)
            self.assertEqual(s["gram_positive_pct"], native.summary.gram_positive_pct)
            self.assertEqual(s["gram_negative_pct"], native.summary.gram_negative_pct)
        else:
            self.assertNotIn("gram_positive", s)

    def synthetic(self):
        image = np.full((220, 280, 3), (170, 235, 245), np.uint8)
        labels = np.zeros(image.shape[:2], np.int32)
        cv2.circle(labels, (60, 60), 12, 7, -1)
        cv2.ellipse(labels, (180, 140), (35, 8), 30, 0, 360, 42, -1)
        labels[10:12, 200:202] = 100
        image[labels == 7] = (150, 40, 110)
        image[labels == 42] = (90, 60, 200)
        image[labels == 100] = (60, 40, 60)
        seg = SegmentationResult(objects_from_labels(labels), labels > 0, labels)
        return image, seg

    def test_full_analysis_matches_author_analyzer_and_renderer(self):
        analyzer = imbp.Analyzer(use_filter=True)
        for method in ("slic", "gmm", "auto"):
            with self.subTest(method=method):
                native = analyzer.analyze(self.image, method)
                result = run_pipeline(self.image, self.settings(segmentation="imbp_" + method, filtering="imbp_rf"))
                self.assert_native(result.classification, native.morphology)
                np.testing.assert_array_equal(result.segmentation.labels, native.segmentation.labels)
                np.testing.assert_array_equal(result.segmentation.mask, native.mask)
                np.testing.assert_array_equal(result.morphology_image_bgr, native.overlay)
                self.assertEqual(result.object_count, result.classification.summary["objects"])

    def test_otsu_and_custom_classification_parameters_match_both_author_rules(self):
        params = ClassificationSettings(noise_min_area=30, ecc_threshold=0.85, red_delta=35)
        for gram, rule in (("imbp_od", "od_red"), ("imbp_rgb", "rgb")):
            result = run_pipeline(self.crop, self.settings(gram_classification=gram, classification_params=params))
            native = imbp.classify_objects(self.crop, result.segmentation.mask, instances=result.segmentation.labels,
                                          noise_min_area=30, ecc_threshold=0.85, red_delta=35, gram_rule=rule)
            self.assert_native(result.classification, native, gram=gram)

    def test_independent_outputs_reuse_labels_without_resegmentation_or_splitting(self):
        image, seg = self.synthetic()
        labels_before = seg.labels.copy()
        for morph in ("none", "imbp_ellipse"):
            for gram in ("none", "imbp_od", "imbp_rgb"):
                with self.subTest(morph=morph, gram=gram):
                    with patch.object(imbp.instances, "split_instances", side_effect=AssertionError("Repeated splitting")):
                        got = classification.classify_segmentation(image, seg, self.settings(morphology=morph, gram_classification=gram))
                    if morph == gram == "none":
                        self.assertIsNone(got)
                        continue
                    native = imbp.classify_objects(image, seg.mask, instances=seg.labels,
                                                  gram_rule="rgb" if gram == "imbp_rgb" else "od_red")
                    self.assert_native(got, native, morph=morph != "none", gram=gram)
                    self.assertEqual({o.object_id for o in got.objects}, {"7", "42", "100"})
        np.testing.assert_array_equal(seg.labels, labels_before)

    def test_synthetic_morphotypes_gram_and_artifact_denominator(self):
        image, seg = self.synthetic()
        result = classification.classify_segmentation(image, seg, self.settings())
        self.assertEqual(result.summary, dict(objects=3, valid=2, artifacts=1, cocci=1, bacilli=1,
                                             gram_positive=1, gram_negative=1, gram_positive_pct=50.0, gram_negative_pct=50.0))
        self.assertIsNone(next(o for o in result.objects if o.artifact).gram)

    def test_light_five_images_preserve_each_contour_id_area_and_color(self):
        for number in range(1, 6):
            with self.subTest(image=number):
                image = cv2.imread(str(ROOT / f"test_input_{number}.bmp"))
                result = run_pipeline(image, PipelineSettings(morphology="imbp_ellipse", gram_classification="imbp_rgb"))
                self.assertIsNone(result.segmentation.mask)
                self.assertIsNone(result.segmentation.labels)
                self.assertEqual(result.object_count, len(result.classification.objects))
                for source, obj in zip(result.segmentation.objects, result.classification.objects):
                    self.assertEqual(obj.object_id, source.object_id)
                    x, y, w, h = cv2.boundingRect(source.contour)
                    mask = np.zeros((h, w), np.uint8)
                    cv2.drawContours(mask, [source.contour - np.array([x, y])], -1, 1, -1)
                    self.assertEqual(obj.area, int(mask.sum()))
                    np.testing.assert_array_equal(obj.mean_bgr, result.processed_image_bgr[y:y+h, x:x+w][mask > 0].mean(axis=0))

    def test_overlapping_light_contours_match_individual_full_image_analysis(self):
        image = self.crop
        contours = (
            cv2.ellipse2Poly((120, 120), (50, 30), 0, 0, 360, 5).reshape(-1, 1, 2),
            cv2.ellipse2Poly((120, 120), (20, 10), 0, 0, 360, 5).reshape(-1, 1, 2),
            cv2.ellipse2Poly((140, 120), (50, 30), 30, 0, 360, 5).reshape(-1, 1, 2),
        )
        seg = SegmentationResult(tuple(DetectedObject(str(i + 1), c) for i, c in enumerate(contours)))
        got = classification.classify_segmentation(image, seg, self.settings())
        for i, obj in enumerate(got.objects, start=1):
            labels = np.zeros(image.shape[:2], np.int32)
            cv2.drawContours(labels, [contours[i - 1]], -1, i, -1)
            native = imbp.classify_objects(image, labels > 0, instances=labels)
            self.assert_native(replace(got, objects=(obj,)), native)

    def test_threshold_equalities_keep_native_decisions(self):
        image, seg = self.synthetic()
        initial = classification.classify_segmentation(image, seg, self.settings())
        circle = next(o for o in initial.objects if o.object_id == "7")
        params = ClassificationSettings(noise_min_area=circle.area, ecc_threshold=circle.eccentricity)
        got = classification.classify_segmentation(image, seg, self.settings(classification_params=params))
        target = next(o for o in got.objects if o.object_id == "7")
        self.assertEqual(target.morphotype, "bacillus")
        self.assertFalse(target.artifact)
        image[seg.labels == 7] = (100, 70, 115)
        rgb = self.settings(gram_classification="imbp_rgb")
        got = classification.classify_segmentation(image, seg, rgb)
        self.assertEqual(next(o for o in got.objects if o.object_id == "7").gram, "positive")
        image[seg.labels == 7] = (100, 70, 116)
        got = classification.classify_segmentation(image, seg, rgb)
        self.assertEqual(next(o for o in got.objects if o.object_id == "7").gram, "negative")

    def test_empty_and_all_artifact_results_have_zero_percentages(self):
        image, seg = self.synthetic()
        params = ClassificationSettings(noise_min_area=100000)
        result = classification.classify_segmentation(image, seg, self.settings(classification_params=params))
        self.assertEqual(result.summary["valid"], 0)
        self.assertEqual(result.summary["artifacts"], 3)
        self.assertEqual(result.summary["gram_positive_pct"], 0)
        empty = run_pipeline(np.full((80, 100, 3), 255, np.uint8), self.settings())
        self.assertEqual(empty.classification.summary["objects"], 0)
        self.assertIn("Нет объектов", classification.classification_text(empty))
        np.testing.assert_array_equal(empty.morphology_image_bgr, empty.processed_image_bgr)

    def test_noncontiguous_preprocessed_input_matches_author_and_is_unchanged(self):
        image = self.crop[:, ::-1]
        before = image.copy()
        result = run_pipeline(image, self.settings(preprocessing="light_average_pool"))
        native = imbp.classify_objects(result.processed_image_bgr, result.segmentation.mask, instances=result.segmentation.labels)
        self.assert_native(result.classification, native)
        np.testing.assert_array_equal(image, before)

    def test_connected_components_remain_connected_components(self):
        result = run_pipeline(self.crop, self.settings(instances="connected_components"))
        self.assertEqual(result.object_count, cv2.connectedComponents(result.segmentation.mask.astype(np.uint8), connectivity=8)[0] - 1)
        self.assertEqual(result.object_count, len(result.classification.objects))
        self.assertIn("Анализ связных областей", classification.classification_text(result))

    def test_disabled_analysis_and_classification_without_rf_models(self):
        with patch.object(classification, "load_imbp", side_effect=AssertionError("Unexpected classification import")):
            result = run_pipeline(self.crop, PipelineSettings())
            self.assertIsNone(result.classification)
            self.assertIsNone(result.morphology_image_bgr)
            self.assertIsNone(result.gram_image_bgr)
        with patch.object(imbp.config, "MODEL_PATH", ROOT / "absent.joblib"), patch.object(imbp.config, "OBJECT_FILTER_PATH", ROOT / "absent.joblib"), patch.object(imbp.config, "METHOD_CHOOSER_PATH", ROOT / "absent.joblib"):
            result = run_pipeline(self.crop, self.settings())
            self.assertIsNotNone(result.classification)

    def test_classification_errors_and_cancellation_propagate(self):
        with patch.object(imbp, "classify_objects", side_effect=RuntimeError("Classification failed")):
            with self.assertRaisesRegex(RuntimeError, "Classification failed"):
                run_pipeline(self.crop, self.settings())
        stopped = False
        native = imbp.classify_objects

        def stop_after(*args, **kwargs):
            nonlocal stopped
            result = native(*args, **kwargs)
            stopped = True
            return result

        with patch.object(imbp, "classify_objects", side_effect=stop_after):
            with self.assertRaises(ProcessingCancelled):
                run_pipeline(self.crop, self.settings(), should_stop=lambda: stopped)

    def test_json_export_preserves_values_settings_and_disabled_outputs(self):
        result = run_pipeline(self.crop, self.settings(morphology="none", gram_classification="imbp_rgb"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "результат.json"
            classification.export_classification(result, path)
            content = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(content["summary"], result.classification.summary)
            self.assertEqual(content["settings"]["classification_params"]["red_delta"], 15)
            self.assertEqual(content["image_shape"], list(self.crop.shape[:2]))
            self.assertEqual(content["objects"][0]["object_id"], result.classification.objects[0].object_id)
            self.assertTrue(all(o["morphotype"] is None and o["od_red_share"] is None for o in content["objects"]))
            with self.assertRaises(ValueError):
                classification.export_classification(replace(result, classification=None), path)

    def test_invalid_classification_settings_are_rejected(self):
        for field, values in (
            ("noise_min_area", (-1, 100001, 1.5, True, None)),
            ("ecc_threshold", (-0.01, 1.01, float("nan"), float("inf"), True, "0.7")),
            ("red_delta", (-1, 256, float("nan"), float("inf"), True, None)),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ClassificationSettings(**{field: value})
        with self.assertRaises(ValueError):
            PipelineSettings(classification_params=None)
        for field in ("morphology", "gram_classification"):
            with self.assertRaises(ValueError):
                PipelineSettings(**{field: "unknown"})


if __name__ == "__main__":
    unittest.main()
