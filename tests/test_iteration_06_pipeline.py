from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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

from Bacteria_finder_pipeline import PipelineSettings, ProcessingCancelled, run_pipeline
from Bacteria_finder_pipeline import imbp_2_resolution as resolution
from Bacteria_finder_pipeline.imbp_classification import export_classification


ROOT = Path(__file__).resolve().parents[1]
NATIVE_RUNNER = '''
import ast
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

source = Path(sys.argv[1])
tree = ast.parse(source.read_text(encoding="utf-8-sig"))
functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
exec(compile(functions, str(source), "exec"), globals())
for k in range(int(sys.argv[2])):
    process_image(k, "input", 2, 2)
'''


class ResolutionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image = cv2.imread(str(ROOT / "test_input_1.bmp"))
        cls.image = image[600:665, 1000:1079].copy()
        y, x = np.indices((23, 31))
        gradient = np.stack(((x * 11 + y * 5) % 256, (x * 7 + y * 13) % 256, (x * 3 + y * 19) % 256), axis=2).astype(np.uint8)
        cls.cases = (cls.image, gradient, np.zeros((9, 13, 3), np.uint8), np.full((1, 1, 3), 255, np.uint8))
        cls.reference = []
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory) / "test_images/input"
            inputs.mkdir(parents=True)
            for k, value in enumerate(cls.cases):
                cv2.imencode(".png", value)[1].tofile(inputs / f"input{k}_jpg.jpg")
            process = subprocess.run(
                [sys.executable, "-c", NATIVE_RUNNER, str(resolution.SOURCE), str(len(cls.cases))],
                cwd=directory, capture_output=True, text=True, timeout=120,
            )
            if process.returncode:
                raise AssertionError(process.stderr)
            for k in range(len(cls.cases)):
                cls.reference.append(cv2.imdecode(np.fromfile(inputs / f"Result_input{k}.jpg", np.uint8), cv2.IMREAD_COLOR))

    def settings(self, **kwargs):
        return replace(PipelineSettings(
            preprocessing="imbp_2_resolution", segmentation="imbp_otsu",
            instances="imbp_watershed", morphology="imbp_ellipse", gram_classification="imbp_od",
        ), **kwargs)

    def test_exact_match_to_author_file_processing_including_jpeg_roundtrips(self):
        for original, expected in zip(self.cases, self.reference):
            with self.subTest(shape=original.shape):
                actual = resolution.enhance_image(original)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertEqual(actual.shape[:2], (original.shape[0] * 2, original.shape[1] * 2))
        lanczos = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        self.assertFalse(np.array_equal(self.reference[0], lanczos))

    def test_vendor_copy_matches_manifest(self):
        manifest = json.loads((ROOT / "third_party/imbp_2/source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_archive_sha256"], "cfaa92f782dfb9d1461511eaa05d72a14d45f1da0023d507cee8d278c8a1b1c4")
        for item in manifest["files"]:
            content = (ROOT / "third_party/imbp_2" / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])

    def test_noncontiguous_input_is_unchanged(self):
        image = self.image[:, ::-1]
        before = image.copy()
        result = resolution.enhance_image(image)
        np.testing.assert_array_equal(result, resolution.enhance_image(before))
        np.testing.assert_array_equal(image, before)
        self.assertFalse(np.shares_memory(result, image))

    def test_concurrent_calls_have_separate_images_and_do_not_change_directory(self):
        with patch("os.chdir", side_effect=AssertionError("Global directory change")):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(resolution.enhance_image, self.cases[:2]))
        for actual, expected in zip(results, self.reference):
            np.testing.assert_array_equal(actual, expected)

    def test_all_segmenters_match_processing_author_output_without_double_preprocessing(self):
        for method in ("light_ridges", "imbp_otsu", "imbp_slic", "imbp_gmm", "imbp_auto"):
            with self.subTest(method=method):
                settings = PipelineSettings(
                    preprocessing="imbp_2_resolution", segmentation=method,
                    filtering="light_area" if method == "light_ridges" else "imbp_rf",
                    instances="light_watershed" if method == "light_ridges" else "imbp_watershed",
                    morphology="imbp_ellipse", gram_classification="imbp_od",
                )
                expected = run_pipeline(self.reference[0], replace(settings, preprocessing="none"))
                with patch.object(resolution, "enhance_image", wraps=resolution.enhance_image) as enhance:
                    actual = run_pipeline(self.image, settings)
                self.assertEqual(enhance.call_count, 1)
                self.assertEqual(actual.source_shape, self.image.shape[:2])
                self.assertEqual(actual.processed_shape, self.reference[0].shape[:2])
                self.assertEqual(actual.coordinate_scale_xy, (2.0, 2.0))
                self.assertEqual(actual.object_count, expected.object_count)
                self.assertEqual(actual.classification, expected.classification)
                np.testing.assert_array_equal(actual.processed_image_bgr, self.reference[0])
                np.testing.assert_array_equal(actual.overlay_image_bgr, expected.overlay_image_bgr)
                np.testing.assert_array_equal(actual.morphology_image_bgr, expected.morphology_image_bgr)
                np.testing.assert_array_equal(actual.gram_image_bgr, expected.gram_image_bgr)
                for attribute in ("labels", "mask", "superpixels", "clusters"):
                    np.testing.assert_array_equal(getattr(actual.segmentation, attribute), getattr(expected.segmentation, attribute))
                for obj, reference in zip(actual.segmentation.objects, expected.segmentation.objects):
                    self.assertEqual(obj.object_id, reference.object_id)
                    np.testing.assert_array_equal(obj.contour, reference.contour)
                    self.assertGreaterEqual(obj.contour.min(), 0)
                    self.assertLess(obj.contour[:, :, 0].max(), actual.processed_shape[1])
                    self.assertLess(obj.contour[:, :, 1].max(), actual.processed_shape[0])

    def test_json_uses_working_coordinates_and_records_original_shape_and_parameters(self):
        result = run_pipeline(self.image, self.settings())
        self.assertGreater(result.object_count, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "анализ.json"
            export_classification(result, path)
            content = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(content["image_shape"], [130, 158])
        self.assertEqual(content["source_image_shape"], [65, 79])
        self.assertEqual(content["coordinate_system"]["reference"], "processed_image")
        self.assertEqual(content["coordinate_system"]["scale_xy_from_source"], [2.0, 2.0])
        self.assertEqual(content["coordinate_system"]["index_base"], 0)
        self.assertEqual(content["coordinate_system"]["area_unit"], "processed_pixel_squared")
        self.assertEqual(content["preprocessing_parameters"], {"gain": 2, "scale_factor": 2})
        self.assertEqual(content["settings"]["classification_params"]["noise_min_area"], 15)
        for obj in content["objects"]:
            self.assertEqual(obj["area"], np.count_nonzero(result.segmentation.labels == int(obj["object_id"])))
            x, y, w, h = obj["bbox"]
            self.assertLessEqual(x + w, 158)
            self.assertLessEqual(y + h, 130)

    def test_other_preprocessing_modes_do_not_load_new_source(self):
        with patch.object(resolution, "_load_functions", side_effect=AssertionError("Unexpected source load")):
            for preprocessing in ("none", "light_average_pool"):
                result = run_pipeline(self.image, self.settings(preprocessing=preprocessing))
                self.assertEqual(result.processed_shape, result.source_shape)
                self.assertEqual(result.coordinate_scale_xy, (1.0, 1.0))

    def test_blank_image_has_matching_shapes_and_no_objects(self):
        result = run_pipeline(np.full((31, 41, 3), 255, np.uint8), self.settings())
        self.assertEqual(result.object_count, 0)
        for image in (result.processed_image_bgr, result.overlay_image_bgr, result.morphology_image_bgr, result.gram_image_bgr):
            self.assertEqual(image.shape, (62, 82, 3))
        self.assertEqual(result.segmentation.mask.shape, (62, 82))

    def test_cancellation_before_processing_does_not_call_enhancement(self):
        with patch.object(resolution, "enhance_image", side_effect=AssertionError("Unexpected work")):
            with self.assertRaises(ProcessingCancelled):
                run_pipeline(self.image, self.settings(), should_stop=lambda: True)

    def test_cancellation_between_channels_prevents_segmentation(self):
        messages = []
        stopped = False

        def progress(message):
            nonlocal stopped
            messages.append(message)
            if "канал 1 из 3" in message:
                stopped = True

        with patch("Bacteria_finder_pipeline.imbp_adapter.ImbpAdapter.segment", side_effect=AssertionError("Unexpected segmentation")):
            with self.assertRaises(ProcessingCancelled):
                run_pipeline(self.image, self.settings(), on_progress=progress, should_stop=lambda: stopped)
        self.assertTrue(any("канал 1 из 3" in message for message in messages))
        self.assertFalse(any("канал 2 из 3" in message for message in messages))

    def test_missing_source_has_actionable_error(self):
        resolution._load_functions.cache_clear()
        try:
            with patch.object(resolution, "SOURCE", ROOT / "missing_resolution.py"):
                with self.assertRaisesRegex(RuntimeError, "Восстановите third_party/imbp_2/resolution/main.py"):
                    resolution.enhance_image(self.image)
        finally:
            resolution._load_functions.cache_clear()

    def test_encoder_failure_is_propagated(self):
        with patch.object(cv2, "imencode", return_value=(False, None)):
            with self.assertRaisesRegex(RuntimeError, "закодировать"):
                resolution.enhance_image(self.image)

    def test_invalid_input_is_rejected(self):
        for image in (None, [], np.empty((0, 4, 3), np.uint8), np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8), self.image.astype(float)):
            with self.subTest(type=type(image)), self.assertRaises(ValueError):
                resolution.enhance_image(image)


if __name__ == "__main__":
    unittest.main()
