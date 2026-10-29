import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from Bacteria_finder_pipeline import OtsuSettings, PipelineSettings, SlicSettings, run_pipeline
from Bacteria_finder_pipeline import imbp_adapter
from Bacteria_finder_pipeline.light_adapter import LightAdapter
from third_party.imbp.src import imbp


ROOT = Path(__file__).resolve().parents[1]


class ImbpIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imdecode(
            np.fromfile(ROOT / "test_input_1.bmp", dtype=np.uint8), cv2.IMREAD_COLOR
        )
        cls.crop = cls.image[700:1105, 500:904].copy()

    def assert_mask(self, result, expected, source):
        segmentation = result.segmentation
        np.testing.assert_array_equal(segmentation.mask, expected)
        self.assertEqual(segmentation.mask.dtype, np.bool_)
        self.assertEqual(segmentation.labels.dtype, np.int32)
        np.testing.assert_array_equal(segmentation.labels > 0, expected)
        count = cv2.connectedComponents(expected.astype(np.uint8), connectivity=8)[0] - 1
        self.assertEqual(result.object_count, count)
        self.assertEqual(segmentation.count_label, "Компоненты маски")
        np.testing.assert_array_equal(result.processed_image_bgr, source)
        contours, _ = cv2.findContours(expected.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        overlay = cv2.drawContours(source.copy(), contours, -1, (0, 255, 0), 1)
        np.testing.assert_array_equal(result.overlay_image_bgr, overlay)

    def test_vendor_sources_and_model_match_archive_hashes(self):
        root = ROOT / "third_party/imbp"
        manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
        for record in manifest["files"]:
            with self.subTest(path=record["path"]):
                self.assertEqual(hashlib.sha256((root / record["path"]).read_bytes()).hexdigest(), record["sha256"])

    def test_otsu_matches_native_on_all_original_images(self):
        for index in range(1, 6):
            with self.subTest(image=index):
                image = cv2.imdecode(np.fromfile(ROOT / f"test_input_{index}.bmp", dtype=np.uint8), cv2.IMREAD_COLOR)
                before = image.copy()
                native = imbp.masks.otsu_bacteria_mask(image)
                result = run_pipeline(image, PipelineSettings(segmentation="imbp_otsu"))
                self.assert_mask(result, native.mask, image)
                np.testing.assert_array_equal(image, before)

    def test_otsu_channels_thresholds_and_noncontiguous_input(self):
        image = self.crop[:, ::-1]
        for channel in ("gray", "flat_gray", "od"):
            for threshold in (None, 75.5, 191):
                with self.subTest(channel=channel, threshold=threshold):
                    native = imbp.masks.otsu_bacteria_mask(image, channel=channel, threshold=threshold)
                    result = run_pipeline(image, PipelineSettings(
                        segmentation="imbp_otsu", otsu=OtsuSettings(channel, threshold)
                    ))
                    self.assert_mask(result, native.mask, image)

    def test_all_slic_modes_preserve_native_full_image_masks_and_superpixels(self):
        predictor = imbp.SlicParamPredictor.load(ROOT / "third_party/imbp/models/slic_params_rf.joblib")
        for mode in ("manual", "analytical", "rf"):
            with self.subTest(mode=mode):
                settings = SlicSettings(mode=mode, k=500, m=30)
                native = imbp.segment_slic(
                    self.image, predictor=predictor if mode == "rf" else None,
                    fixed_params=(500, 30) if mode == "manual" else None,
                    use_analytical=mode == "analytical",
                )
                result = run_pipeline(self.image, PipelineSettings(segmentation="imbp_slic", slic=settings))
                self.assert_mask(result, native.mask, self.image)
                np.testing.assert_array_equal(result.segmentation.superpixels, native.labels)
                self.assertEqual(
                    [(t.bbox, t.k, t.m, t.source) for t in result.segmentation.tiles],
                    [(t.bbox, t.k, t.m, t.source) for t in native.tiles],
                )
                self.assertTrue(result.segmentation.tiles)
                self.assertNotEqual(result.object_count, len(np.unique(native.labels)) - 1)

    def test_implied_defaults_and_incompatible_stage_choices(self):
        for method in ("imbp_otsu", "imbp_slic"):
            settings = PipelineSettings(segmentation=method)
            self.assertEqual(settings.preprocessing, "none")
            self.assertEqual(settings.filtering, "imbp_native")
            self.assertEqual(settings.instances, "connected_components")
            for wrong in ({"filtering": "light_area"}, {"instances": "light_watershed"}):
                with self.subTest(method=method, wrong=wrong), self.assertRaises(ValueError):
                    PipelineSettings(segmentation=method, **wrong)
        with self.assertRaises(ValueError):
            PipelineSettings(instances="connected_components")

    def test_explicit_average_pooling_is_the_only_input_adaptation(self):
        expected = LightAdapter().prepare(self.crop, "light_average_pool")
        result = run_pipeline(self.crop, PipelineSettings(segmentation="imbp_otsu", preprocessing="light_average_pool"))
        np.testing.assert_array_equal(result.processed_image_bgr, expected)
        np.testing.assert_array_equal(result.segmentation.mask, imbp.masks.otsu_bacteria_mask(expected).mask)
        np.testing.assert_array_equal(run_pipeline(self.crop, PipelineSettings(preprocessing="none")).processed_image_bgr, self.crop)

    def test_component_mapping_preserves_holes_tiny_regions_and_diagonal_contact(self):
        mask = np.zeros((40, 40), bool)
        mask[2:20, 2:20] = True
        mask[5:17, 5:17] = False
        mask[25, 25] = True
        mask[26, 26] = True
        mask[35, 3] = True
        result = imbp_adapter.mask_result(mask)
        np.testing.assert_array_equal(result.mask, mask)
        np.testing.assert_array_equal(result.labels > 0, mask)
        self.assertEqual(len(result.objects), 3)
        self.assertEqual(result.labels[25, 25], result.labels[26, 26])
        mask[:] = False
        self.assertTrue(result.mask.any())

    def test_uniform_white_image_is_empty_in_all_modes(self):
        image = np.full((65, 97, 3), 255, np.uint8)
        settings = [PipelineSettings(segmentation="imbp_otsu")]
        settings += [PipelineSettings(segmentation="imbp_slic", slic=SlicSettings(mode=mode)) for mode in ("manual", "analytical", "rf")]
        for item in settings:
            with self.subTest(settings=item):
                result = run_pipeline(image, item)
                self.assertEqual(result.object_count, 0)
                self.assertFalse(result.segmentation.mask.any())
                np.testing.assert_array_equal(result.overlay_image_bgr, image)

    def test_invalid_parameters_are_rejected(self):
        values = [
            lambda: SlicSettings(mode="other"), lambda: SlicSettings(k=0),
            lambda: SlicSettings(k=3.5), lambda: SlicSettings(k=True),
            lambda: SlicSettings(m=0), lambda: SlicSettings(m=float("nan")),
            lambda: OtsuSettings(channel="other"), lambda: OtsuSettings(threshold=256),
            lambda: OtsuSettings(threshold=True), lambda: PipelineSettings(slic={}),
        ]
        for make_settings in values:
            with self.assertRaises(ValueError):
                make_settings()

    def test_missing_rf_model_does_not_fallback_and_other_modes_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.joblib"
            with patch.object(imbp.config, "MODEL_PATH", missing):
                with self.assertRaisesRegex(RuntimeError, "RF-модель SLIC недоступна"):
                    run_pipeline(self.crop, PipelineSettings(segmentation="imbp_slic"))
                for mode in ("manual", "analytical"):
                    result = run_pipeline(self.crop, PipelineSettings(segmentation="imbp_slic", slic=SlicSettings(mode=mode)))
                    self.assertIsNotNone(result.segmentation.mask)

    def test_corrupt_model_and_wrong_sklearn_version_are_reported(self):
        import sklearn
        imbp_adapter._load_predictor.cache_clear()
        with patch.object(sklearn, "__version__", "0.0"):
            with self.assertRaisesRegex(RuntimeError, "scikit-learn==1.9.1"):
                imbp_adapter.load_slic_predictor()
        with tempfile.TemporaryDirectory() as directory:
            damaged = Path(directory) / "bad.joblib"
            damaged.write_bytes(b"invalid model")
            with patch.object(imbp.config, "MODEL_PATH", damaged):
                with self.assertRaisesRegex(RuntimeError, "Не удалось загрузить RF-модель"):
                    imbp_adapter.load_slic_predictor()

    def test_model_cache_reuses_loaded_model(self):
        first = imbp_adapter.load_slic_predictor()
        self.assertIs(first, imbp_adapter.load_slic_predictor())

    def test_light_import_does_not_require_imbp_dependencies(self):
        script = '''
import importlib.abc, sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ('sklearn', 'pandas', 'joblib', 'third_party'):
            raise ImportError('Blocked optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
import numpy as np
from Bacteria_finder_pipeline import run_pipeline
assert run_pipeline(np.full((64,64,3),255,np.uint8)).object_count == 0
'''
        completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
