from copy import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

from Bacteria_finder_pipeline import (
    InstanceSettings, ObjectFilterSettings, PipelineSettings, ProcessingCancelled, run_pipeline,
)
from Bacteria_finder_pipeline import imbp_postprocessing as post
from Bacteria_finder_pipeline.imbp_adapter import mask_result
from Bacteria_finder_pipeline.light_adapter import LightAdapter
from third_party.imbp.src import imbp


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("imbp_otsu", "imbp_slic", "imbp_gmm", "imbp_auto")


class ImbpPostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imdecode(np.fromfile(ROOT / "test_input_1.bmp", dtype=np.uint8), cv2.IMREAD_COLOR)
        cls.crop = cls.image[700:981, 500:789].copy()
        cls.model = imbp.object_filter.ObjectFilter.load(ROOT / "third_party/imbp/models/object_filter_rf.joblib")

    def assert_instances(self, result, mask, labels):
        seg = result.segmentation
        np.testing.assert_array_equal(seg.mask, mask)
        np.testing.assert_array_equal(seg.labels, labels)
        self.assertEqual({int(obj.object_id) for obj in seg.objects}, set(np.unique(labels)) - {0})
        self.assertEqual(result.object_count, len(np.unique(labels[labels > 0])))
        self.assertEqual(seg.count_label, "Клетки")
        expected = self.image.copy() if mask.shape == self.image.shape[:2] else None
        if expected is not None:
            expected[find_boundaries(labels, connectivity=1, mode="inner")] = (0, 255, 0)
            np.testing.assert_array_equal(result.overlay_image_bgr, expected)

    def test_full_pipeline_matches_author_analyzer_on_original_image(self):
        analyzer = imbp.Analyzer(use_filter=True)
        for method in ("slic", "gmm", "auto"):
            with self.subTest(method=method):
                native = analyzer.segment(self.image, method)
                result = run_pipeline(self.image, PipelineSettings(
                    segmentation="imbp_" + method, filtering="imbp_rf", instances="imbp_watershed"
                ))
                self.assert_instances(result, native.mask, native.labels)
                self.assertEqual(result.segmentation.n_rejected, native.n_rejected)
                if method == "slic":
                    np.testing.assert_array_equal(result.segmentation.superpixels, native.superpixels)
                if method == "auto":
                    self.assertEqual([(t.bbox, t.method) for t in result.segmentation.tile_methods], native.tile_methods)

    def test_otsu_filter_and_splitting_match_author_functions(self):
        raw = imbp.masks.otsu_bacteria_mask(self.image)
        kept, probabilities = self.model.filter_mask(self.image, raw.mask)
        labels = imbp.instances.split_instances(self.image, kept)
        result = run_pipeline(self.image, PipelineSettings(
            segmentation="imbp_otsu", filtering="imbp_rf", instances="imbp_watershed"
        ))
        self.assert_instances(result, kept, labels)
        np.testing.assert_allclose(result.segmentation.filter_probabilities, probabilities, atol=1e-12)
        self.assertEqual(result.segmentation.filter_threshold, 0.1)

    def test_independent_stage_combinations_preserve_masks_and_metadata(self):
        for method in METHODS:
            raw = run_pipeline(self.crop, PipelineSettings(segmentation=method)).segmentation
            for filtering, splitting in ((True, False), (False, True), (True, True)):
                with self.subTest(method=method, filtering=filtering, splitting=splitting):
                    mask = self.model.filter_mask(self.crop, raw.mask)[0] if filtering else raw.mask
                    result = run_pipeline(self.crop, PipelineSettings(
                        segmentation=method, filtering="imbp_rf" if filtering else "imbp_native",
                        instances="imbp_watershed" if splitting else "connected_components",
                    ))
                    np.testing.assert_array_equal(result.segmentation.mask, mask)
                    self.assertEqual(result.segmentation.tiles, raw.tiles)
                    self.assertEqual(result.segmentation.tile_methods, raw.tile_methods)
                    if splitting:
                        self.assert_instances(result, mask, imbp.instances.split_instances(self.crop, mask))
                    else:
                        self.assertEqual(result.object_count, cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
                    if raw.clusters is not None:
                        np.testing.assert_array_equal(result.segmentation.clusters, raw.clusters)
                    if raw.superpixels is not None:
                        np.testing.assert_array_equal(result.segmentation.superpixels, np.where(mask, raw.superpixels, 0))

    def test_manual_threshold_uses_native_comparison_without_mutating_cached_model(self):
        raw = mask_result(imbp.masks.otsu_bacteria_mask(self.crop).mask)
        cached = post.load_object_filter()
        for threshold in (0, 0.5, 1):
            with self.subTest(threshold=threshold):
                expected, probabilities = replace(self.model, threshold=threshold).filter_mask(self.crop, raw.mask)
                result = post.filter_objects(self.crop, raw, ObjectFilterSettings(threshold))
                np.testing.assert_array_equal(result.mask, expected)
                np.testing.assert_allclose(result.filter_probabilities, probabilities, atol=1e-12)
                self.assertEqual(result.n_rejected, int((probabilities < threshold).sum()))
                self.assertEqual(cached.threshold, 0.1)
        self.assertEqual(post.filter_objects(self.crop, raw, ObjectFilterSettings()).filter_threshold, 0.1)

    def test_preprocessing_noncontiguous_input_and_custom_split_parameters(self):
        image = self.crop[:, ::-1]
        before = image.copy()
        processed = LightAdapter().prepare(image, "light_average_pool")
        raw = imbp.masks.otsu_bacteria_mask(processed).mask
        kept, _ = self.model.filter_mask(processed, raw)
        params = InstanceSettings(h=2, sigma=1, beta=0, neck_ratio=0)
        expected = imbp.instances.split_instances(processed, kept, h=2, sigma=1, beta=0, neck_ratio=0)
        result = run_pipeline(image, PipelineSettings(
            segmentation="imbp_otsu", preprocessing="light_average_pool", filtering="imbp_rf",
            instances="imbp_watershed", instance_params=params,
        ))
        self.assert_instances(result, kept, expected)
        np.testing.assert_array_equal(result.processed_image_bgr, processed)
        np.testing.assert_array_equal(image, before)

    def test_touching_cells_are_split_and_binary_mask_is_preserved(self):
        image = np.full((120, 200, 3), (170, 235, 245), np.uint8)
        for cx in (60, 82, 104):
            cv2.circle(image, (cx, 60), 13, (180, 90, 150), -1)
        for cx in (60, 82, 104):
            cv2.circle(image, (cx, 60), 6, (150, 40, 110), -1)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        mask = imbp.masks.otsu_bacteria_mask(image, channel="gray").mask
        raw = mask_result(mask)
        result = post.split_objects(image, raw, InstanceSettings())
        self.assertEqual(len(raw.objects), 1)
        self.assertEqual(len(result.objects), 3)
        np.testing.assert_array_equal(result.mask, mask)
        np.testing.assert_array_equal(result.labels, imbp.instances.split_instances(image, mask))
        np.testing.assert_array_equal(raw.labels > 0, mask)

    def test_object_ids_are_original_label_ids_including_gaps(self):
        labels = np.zeros((50, 60), np.int32)
        labels[4:10, 7:16] = 1
        labels[25:45, 31:51] = 4
        objects = post.objects_from_labels(labels)
        self.assertEqual([obj.object_id for obj in objects], ["1", "4"])
        for obj in objects:
            for x, y in obj.contour[:, 0, :]:
                self.assertEqual(labels[y, x], int(obj.object_id))

    def test_empty_mask_with_both_stages_has_no_objects(self):
        image = np.full((65, 97, 3), 255, np.uint8)
        result = run_pipeline(image, PipelineSettings(
            segmentation="imbp_otsu", filtering="imbp_rf", instances="imbp_watershed"
        ))
        self.assertEqual(result.object_count, 0)
        self.assertEqual(result.segmentation.n_rejected, 0)
        self.assertEqual(result.segmentation.filter_probabilities.size, 0)
        np.testing.assert_array_equal(result.overlay_image_bgr, image)

    def test_full_foreground_is_rejected_before_unsafe_author_call(self):
        image = np.full((32, 32, 3), 200, np.uint8)
        raw = mask_result(np.ones(image.shape[:2], bool))
        with patch.object(imbp.instances, "split_instances", side_effect=AssertionError("unsafe call")) as native:
            with self.assertRaisesRegex(ValueError, "без фона"):
                post.split_objects(image, raw, InstanceSettings())
            native.assert_not_called()

    def test_missing_or_invalid_model_reports_error_and_unfiltered_mode_still_works(self):
        import sklearn

        post._load_object_filter.cache_clear()
        with patch.object(sklearn, "__version__", "0.0"):
            with self.assertRaisesRegex(RuntimeError, "scikit-learn==1.9.1"):
                post.load_object_filter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filter.joblib"
            with patch.object(imbp.config, "OBJECT_FILTER_PATH", path):
                with self.assertRaisesRegex(RuntimeError, "RF-фильтр недоступен"):
                    run_pipeline(self.crop, PipelineSettings(segmentation="imbp_otsu", filtering="imbp_rf"))
                self.assertIsNotNone(run_pipeline(self.crop, PipelineSettings(segmentation="imbp_otsu", instances="imbp_watershed")))
                path.write_bytes(b"broken model")
                with self.assertRaisesRegex(RuntimeError, "Не удалось загрузить RF-фильтр"):
                    post.load_object_filter()
        wrong = copy(self.model)
        wrong.meta = dict(self.model.meta, features=list(reversed(self.model.meta["features"])))
        post._load_object_filter.cache_clear()
        with patch.object(imbp.object_filter.ObjectFilter, "load", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "Не удалось загрузить RF-фильтр"):
                post.load_object_filter()

    def test_invalid_parameters_and_light_combinations_are_rejected(self):
        for threshold in (-0.1, 1.1, float("nan"), True, "0.1"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                ObjectFilterSettings(threshold)
        for values in ({"h": 0}, {"sigma": -1}, {"beta": float("inf")}, {"neck_ratio": -1}, {"h": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                InstanceSettings(**values)
        for values in ({"filtering": "imbp_rf"}, {"instances": "imbp_watershed"}, {"object_filter": {}}, {"instance_params": {}}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PipelineSettings(**values)

    def test_stage_order_and_cancellation_after_filter_before_splitting(self):
        phases = []
        stopped = []
        native_filter = post.filter_objects

        def filtered(*args):
            result = native_filter(*args)
            stopped.append(True)
            return result

        with patch("Bacteria_finder_pipeline.pipeline.filter_objects", side_effect=filtered), patch("Bacteria_finder_pipeline.pipeline.split_objects", side_effect=AssertionError("unexpected splitting")):
            with self.assertRaises(ProcessingCancelled):
                run_pipeline(self.crop, PipelineSettings(
                    segmentation="imbp_otsu", filtering="imbp_rf", instances="imbp_watershed"
                ), on_progress=phases.append, should_stop=lambda: bool(stopped))
        self.assertEqual(phases[-1], "Фильтрация ложных объектов (RF)…")
        self.assertIn("Сегментация…", phases)

    def test_invalid_author_label_coverage_is_reported_without_modifying_it(self):
        raw = mask_result(imbp.masks.otsu_bacteria_mask(self.crop).mask)
        self.assertTrue(raw.mask.any())
        with patch.object(imbp.instances, "split_instances", return_value=np.zeros(raw.mask.shape, np.int32)):
            with self.assertRaisesRegex(ValueError, "не совпадающие"):
                post.split_objects(self.crop, raw, InstanceSettings())


if __name__ == "__main__":
    unittest.main()
