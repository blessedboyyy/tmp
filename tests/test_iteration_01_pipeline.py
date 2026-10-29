from pathlib import Path
import unittest

import cv2
import numpy as np

from Bacteria_finder_core_light.BFmainlib import BF_image
from Bacteria_finder_pipeline import PipelineSettings, ProcessingCancelled, run_pipeline


ROOT = Path(__file__).resolve().parents[1]


class PipelineRegressionTests(unittest.TestCase):
    def assert_matches_original(self, image, sigma=2.5):
        before = image.copy()
        original = BF_image(verbose=False)
        original.load_image(how="image", image=image)
        original.preprocess_image()
        original.segment_image(type="ridges", sigma=sigma)
        original_objects = list(original.objects_db.values())
        expected_overlay = cv2.drawContours(
            image.copy(), [obj.object_countour_coords for obj in original_objects],
            -1, (0, 255, 0), 1,
        )
        result = run_pipeline(image, PipelineSettings(sigma=sigma))
        self.assertEqual(len(original_objects), result.object_count)
        self.assertEqual(result.source_shape, image.shape[:2])
        for expected, actual in zip(original_objects, result.segmentation.objects):
            self.assertEqual(expected.object_id, actual.object_id)
            np.testing.assert_array_equal(expected.object_countour_coords, actual.contour)
        np.testing.assert_array_equal(expected_overlay, result.overlay_image_bgr)
        np.testing.assert_array_equal(
            original.bacteria_image_preprocessed[:, :, ::-1], result.processed_image_bgr
        )
        np.testing.assert_array_equal(before, image)
        self.assertTrue(result.processed_image_bgr.flags.c_contiguous)
        self.assertIsNone(result.segmentation.mask)
        self.assertIsNone(result.segmentation.labels)

    def test_original_images_match_exactly(self):
        for index in range(1, 6):
            with self.subTest(image=f"test_input_{index}.bmp"):
                source = ROOT / f"test_input_{index}.bmp"
                image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(image)
                self.assert_matches_original(image)

    def test_sigma_boundaries_and_noncontiguous_odd_image(self):
        image = cv2.imdecode(
            np.fromfile(ROOT / "test_input_1.bmp", dtype=np.uint8), cv2.IMREAD_COLOR
        )[:257, :385, :][:, ::-1, :]
        for sigma in (0.1, 1.0, 20.0):
            with self.subTest(sigma=sigma):
                self.assert_matches_original(image, sigma)

    def test_uniform_images_have_no_objects(self):
        for value in (0, 255):
            with self.subTest(value=value):
                image = np.full((64, 64, 3), value, dtype=np.uint8)
                result = run_pipeline(image)
                self.assertEqual(result.object_count, 0)
                np.testing.assert_array_equal(result.overlay_image_bgr, image)

    def test_invalid_input_is_rejected(self):
        for image in (None, np.zeros((0, 5, 3), np.uint8), np.zeros((5, 5), np.uint8),
                      np.zeros((5, 5, 4), np.uint8), np.zeros((5, 5, 3), np.float64)):
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(ValueError):
                    run_pipeline(image)

    def test_unknown_methods_and_invalid_sigma_are_rejected(self):
        with self.assertRaises(ValueError):
            PipelineSettings(segmentation="unavailable")
        for sigma in (0, 20.1, float("nan"), float("inf"), "2.5", True):
            with self.subTest(sigma=sigma):
                with self.assertRaises(ValueError):
                    PipelineSettings(sigma=sigma)

    def test_cancellation_before_processing(self):
        with self.assertRaises(ProcessingCancelled):
            run_pipeline(np.zeros((64, 64, 3), np.uint8), should_stop=lambda: True)


if __name__ == "__main__":
    unittest.main()
