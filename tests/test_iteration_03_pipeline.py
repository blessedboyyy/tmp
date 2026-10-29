from copy import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from Bacteria_finder_pipeline import GmmSettings, OtsuSettings, PipelineSettings, SlicSettings, run_pipeline
from Bacteria_finder_pipeline import imbp_adapter
from Bacteria_finder_pipeline.light_adapter import LightAdapter
from third_party.imbp.src import imbp
from third_party.imbp.src.imbp.segmentation import fusion


ROOT = Path(__file__).resolve().parents[1]


class GmmAutoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imdecode(
            np.fromfile(ROOT / "test_input_1.bmp", dtype=np.uint8), cv2.IMREAD_COLOR
        )
        cls.crop = cls.image[700:981, 500:789].copy()
        cls.predictor = imbp.SlicParamPredictor.load(ROOT / "third_party/imbp/models/slic_params_rf.joblib")
        cls.chooser = fusion.MethodChooser.load(ROOT / "third_party/imbp/models/method_chooser_rf.joblib")

    def assert_mask(self, result, native, image):
        np.testing.assert_array_equal(result.segmentation.mask, native.mask)
        np.testing.assert_array_equal(result.segmentation.labels > 0, native.mask)
        np.testing.assert_array_equal(result.processed_image_bgr, image)
        expected = cv2.connectedComponents(native.mask.astype(np.uint8), connectivity=8)[0] - 1
        self.assertEqual(result.object_count, expected)
        self.assertEqual(result.segmentation.count_label, "Компоненты маски")
        contours, _ = cv2.findContours(native.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        np.testing.assert_array_equal(
            result.overlay_image_bgr, cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 1)
        )

    def test_gmm_matches_native_on_five_original_images(self):
        for index in range(1, 6):
            with self.subTest(image=index):
                image = cv2.imdecode(np.fromfile(ROOT / f"test_input_{index}.bmp", dtype=np.uint8), cv2.IMREAD_COLOR)
                before = image.copy()
                native = imbp.segment_gmm(image)
                result = run_pipeline(image, PipelineSettings(segmentation="imbp_gmm"))
                self.assert_mask(result, native, image)
                np.testing.assert_array_equal(result.segmentation.clusters, native.clusters)
                np.testing.assert_allclose(result.segmentation.cluster_means, native.cluster_means, atol=1e-10, rtol=1e-10)
                self.assertEqual(result.segmentation.bacteria_clusters, native.bacteria_clusters)
                np.testing.assert_array_equal(image, before)

    def test_auto_matches_native_on_five_original_images(self):
        for index in range(1, 6):
            with self.subTest(image=index):
                image = cv2.imdecode(np.fromfile(ROOT / f"test_input_{index}.bmp", dtype=np.uint8), cv2.IMREAD_COLOR)
                before = image.copy()
                native = fusion.segment_auto(image, predictor=self.predictor, chooser=self.chooser)
                result = run_pipeline(image, PipelineSettings(segmentation="imbp_auto"))
                self.assert_mask(result, native, image)
                self.assertEqual(
                    [(t.bbox, t.method) for t in result.segmentation.tile_methods], native.tile_methods
                )
                np.testing.assert_array_equal(image, before)
                self.assertIsNone(result.segmentation.superpixels)
                self.assertIsNone(result.segmentation.clusters)

    def test_custom_gmm_parameters_and_noncontiguous_input_match_native(self):
        image = self.crop[:, ::-1]
        params = GmmSettings(n_components=4, min_area=0, bilateral=False, hmrf_beta=2.0)
        native = imbp.segment_gmm(image, n_components=4, min_area=0, bilateral=False, hmrf_beta=2.0)
        result = run_pipeline(image, PipelineSettings(segmentation="imbp_gmm", gmm=params))
        self.assert_mask(result, native, image)
        np.testing.assert_array_equal(result.segmentation.clusters, native.clusters)
        self.assertEqual(result.segmentation.cluster_means.shape, (4, 3))

    def test_auto_uses_native_candidates_independent_of_other_method_settings(self):
        native = fusion.segment_auto(self.crop, predictor=self.predictor, chooser=self.chooser)
        result = run_pipeline(self.crop, PipelineSettings(
            segmentation="imbp_auto", otsu=OtsuSettings(channel="od", threshold=250),
            slic=SlicSettings(mode="manual", k=1, m=999),
            gmm=GmmSettings(n_components=8, min_area=100000, bilateral=False, hmrf_beta=50),
        ))
        self.assert_mask(result, native, self.crop)
        self.assertEqual([(t.bbox, t.method) for t in result.segmentation.tile_methods], native.tile_methods)

    def test_explicit_average_pooling_is_passed_to_each_new_method(self):
        image = LightAdapter().prepare(self.crop, "light_average_pool")
        for method in ("imbp_gmm", "imbp_auto"):
            with self.subTest(method=method):
                native = imbp.segment_gmm(image) if method == "imbp_gmm" else fusion.segment_auto(image, predictor=self.predictor, chooser=self.chooser)
                result = run_pipeline(self.crop, PipelineSettings(segmentation=method, preprocessing="light_average_pool"))
                np.testing.assert_array_equal(result.processed_image_bgr, image)
                np.testing.assert_array_equal(result.segmentation.mask, native.mask)

    def test_auto_tile_bounds_cover_an_odd_image_exactly_once(self):
        image = cv2.resize(self.image, (809, 611))[:, ::-1]
        native = fusion.segment_auto(image, predictor=self.predictor, chooser=self.chooser)
        result = run_pipeline(image, PipelineSettings(segmentation="imbp_auto"))
        self.assert_mask(result, native, image)
        covered = np.zeros(image.shape[:2], np.uint8)
        for tile in result.segmentation.tile_methods:
            y0, y1, x0, x1 = tile.bbox
            covered[y0:y1, x0:x1] += 1
            self.assertIn(tile.method, ("otsu", "slic", "gmm"))
        np.testing.assert_array_equal(covered, np.ones(image.shape[:2], np.uint8))

    def test_uniform_white_image_is_empty_and_keeps_native_auto_choices(self):
        image = np.full((65, 97, 3), 255, np.uint8)
        for method in ("imbp_gmm", "imbp_auto"):
            with self.subTest(method=method):
                result = run_pipeline(image, PipelineSettings(segmentation=method))
                self.assertEqual(result.object_count, 0)
                self.assertFalse(result.segmentation.mask.any())
                np.testing.assert_array_equal(result.overlay_image_bgr, image)
                if method == "imbp_auto":
                    self.assertEqual(result.segmentation.tile_methods[0].method, "otsu")
                else:
                    self.assertEqual(result.segmentation.bacteria_clusters, ())

    def test_new_methods_reject_incompatible_stages_and_invalid_parameters(self):
        for method in ("imbp_gmm", "imbp_auto"):
            settings = PipelineSettings(segmentation=method)
            self.assertEqual((settings.preprocessing, settings.filtering, settings.instances), ("none", "imbp_native", "connected_components"))
            with self.assertRaises(ValueError):
                PipelineSettings(segmentation=method, instances="light_watershed")
            with self.assertRaises(ValueError):
                PipelineSettings(segmentation=method, filtering="light_area")
        invalid = [
            {"n_components": 1}, {"n_components": 9}, {"n_components": True}, {"n_components": 3.5},
            {"min_area": -1}, {"min_area": 1.2}, {"min_area": False},
            {"bilateral": 1}, {"hmrf_beta": -0.1}, {"hmrf_beta": float("nan")},
            {"hmrf_beta": float("inf")}, {"hmrf_beta": True},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GmmSettings(**values)
        with self.assertRaises(ValueError):
            PipelineSettings(gmm={})

    def test_missing_models_fail_auto_but_gmm_needs_no_saved_models(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.joblib"
            with patch.object(imbp.config, "METHOD_CHOOSER_PATH", missing):
                with self.assertRaisesRegex(RuntimeError, "Модель выбора метода недоступна"):
                    run_pipeline(self.crop, PipelineSettings(segmentation="imbp_auto"))
            with patch.object(imbp.config, "MODEL_PATH", missing):
                with self.assertRaisesRegex(RuntimeError, "RF-модель SLIC недоступна"):
                    run_pipeline(self.crop, PipelineSettings(segmentation="imbp_auto"))
                with patch.object(imbp.config, "METHOD_CHOOSER_PATH", missing):
                    result = run_pipeline(self.crop, PipelineSettings(segmentation="imbp_gmm"))
                    self.assertIsNotNone(result.segmentation.mask)

    def test_corrupt_or_incompatible_chooser_is_reported_without_fallback(self):
        import sklearn

        imbp_adapter._load_chooser.cache_clear()
        with patch.object(sklearn, "__version__", "0.0"):
            with self.assertRaisesRegex(RuntimeError, "scikit-learn==1.9.1"):
                imbp_adapter.load_method_chooser()
        with tempfile.TemporaryDirectory() as directory:
            damaged = Path(directory) / "bad.joblib"
            damaged.write_bytes(b"invalid model")
            with patch.object(imbp.config, "METHOD_CHOOSER_PATH", damaged):
                with self.assertRaisesRegex(RuntimeError, "Не удалось загрузить модель выбора"):
                    imbp_adapter.load_method_chooser()
        for key, value in (("methods", ["gmm", "otsu", "slic"]), ("features", list(reversed(self.chooser.feature_names)))):
            bad = copy(self.chooser)
            bad.meta = dict(self.chooser.meta, **{key: value})
            imbp_adapter._load_chooser.cache_clear()
            with patch.object(fusion.MethodChooser, "load", return_value=bad):
                with self.assertRaisesRegex(RuntimeError, "Не удалось загрузить модель выбора"):
                    imbp_adapter.load_method_chooser()

    def test_no_object_filter_or_cell_splitting_is_run(self):
        with patch.object(imbp.object_filter.ObjectFilter, "load_or_none", side_effect=AssertionError("unexpected filter")), patch.object(imbp.instances, "split_instances", side_effect=AssertionError("unexpected splitting")):
            for method in ("imbp_gmm", "imbp_auto"):
                result = run_pipeline(self.crop, PipelineSettings(segmentation=method))
                self.assertIsNotNone(result.segmentation.mask)


if __name__ == "__main__":
    unittest.main()
