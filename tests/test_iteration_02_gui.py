import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

import test_iteration_01_gui as previous
from third_party.imbp.src import imbp


class ImbpGuiTests(unittest.TestCase):
    setUpClass = classmethod(previous.GuiIntegrationTests.setUpClass.__func__)
    setUp = previous.GuiIntegrationTests.setUp
    tearDown = previous.GuiIntegrationTests.tearDown
    wait_for = previous.GuiIntegrationTests.wait_for
    load = previous.GuiIntegrationTests.load
    process = previous.GuiIntegrationTests.process

    def select_method(self, method):
        combo = self.window.stage_combos["segmentation"]
        combo.setCurrentIndex(combo.findData(method))
        self.app.processEvents()

    def select_mode(self, mode):
        combo = self.window.combo_slic_mode
        combo.setCurrentIndex(combo.findData(mode))

    def test_switching_methods_restores_compatible_stages_and_preprocessing(self):
        self.load()
        self.process()
        self.window.spin_sigma.setValue(3.7)
        self.select_method("imbp_otsu")
        self.assertIsNone(self.window.result)
        self.assertEqual(self.window.current_settings().preprocessing, "none")
        self.assertEqual(self.window.current_settings().filtering, "imbp_native")
        self.assertEqual(self.window.current_settings().instances, "connected_components")
        self.assertTrue(self.window.gb_otsu.isVisible())
        self.assertFalse(self.window.gb_light.isVisible())
        prep = self.window.stage_combos["preprocessing"]
        prep.setCurrentIndex(prep.findData("light_average_pool"))
        self.select_method("imbp_slic")
        self.assertEqual(self.window.current_settings().preprocessing, "none")
        self.select_method("imbp_otsu")
        self.assertEqual(self.window.current_settings().preprocessing, "light_average_pool")
        self.select_method("light_ridges")
        self.assertEqual(self.window.current_settings().sigma, 3.7)
        self.assertEqual(self.window.current_settings().instances, "light_watershed")
        self.assertFalse(self.window.rb_mask.isVisible())
        self.process()
        self.assertIsNone(self.window.mask_image)

    def test_all_new_modes_run_in_worker_with_correct_result_kind(self):
        self.load()
        for method, mode in (("imbp_otsu", "rf"), ("imbp_slic", "manual"), ("imbp_slic", "analytical"), ("imbp_slic", "rf")):
            with self.subTest(method=method, mode=mode):
                self.select_method(method)
                self.select_mode(mode)
                self.window.segment_image()
                self.assertFalse(self.window.combo_slic_mode.isEnabled())
                self.assertFalse(self.window.btn_save.isEnabled())
                self.assertFalse(self.window.stage_combos["segmentation"].isEnabled())
                self.wait_for(lambda: self.window._worker is None)
                self.window.show_error.assert_not_called()
                self.assertIsNotNone(self.window.result)
                self.assertTrue(self.window.lbl_count.text().startswith("Компоненты маски:"))
                self.assertTrue(self.window.rb_mask.isEnabled())
                self.assertEqual(self.window.result_thread, self.app.thread())
                self.assertTrue(self.window.lbl_details.text())

    def test_manual_slic_parameters_and_otsu_threshold_enable_and_invalidate(self):
        self.load()
        self.select_method("imbp_slic")
        self.assertFalse(self.window.spin_slic_k.isEnabled())
        self.select_mode("manual")
        self.assertTrue(self.window.spin_slic_k.isEnabled())
        self.window.spin_slic_k.setValue(750)
        self.window.spin_slic_m.setValue(20)
        self.process()
        self.assertTrue(self.window.result.segmentation.tiles)
        for tile in self.window.result.segmentation.tiles:
            self.assertEqual((tile.k, tile.m, tile.source), (750, 20, "fixed"))
        self.window.spin_slic_m.setValue(30)
        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window.mask_image)
        self.assertFalse(self.window.rb_mask.isEnabled())
        self.assertTrue(self.window.rb_orig.isChecked())
        self.select_method("imbp_otsu")
        self.assertFalse(self.window.spin_otsu_threshold.isEnabled())
        self.window.check_otsu_auto.setChecked(False)
        self.assertTrue(self.window.spin_otsu_threshold.isEnabled())
        self.window.spin_otsu_threshold.setValue(190)
        self.process()
        native = imbp.masks.otsu_bacteria_mask(self.image, threshold=190)
        np.testing.assert_array_equal(self.window.result.segmentation.mask, native.mask)
        self.window.combo_otsu_channel.setCurrentIndex(1)
        self.assertIsNone(self.window.result)

    def test_mask_preview_and_lossless_unicode_export(self):
        self.load()
        self.select_method("imbp_otsu")
        self.process()
        self.window.rb_mask.setChecked(True)
        expected = self.window.result.segmentation.mask.astype(np.uint8) * 255
        np.testing.assert_array_equal(self.window.current_image(), expected)
        self.assertFalse(self.window.lbl_image.pixmap().isNull())
        for extension in ("png", "bmp"):
            target = self.directory / f"маска.{extension}"
            with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
                self.window.save_image()
            decoded = cv2.imdecode(np.fromfile(target, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, expected)
        self.window.show_error.assert_not_called()
        target = self.directory / "маска.jpg"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
            self.window.save_image()
        self.window.show_error.assert_called_once()
        self.assertFalse(target.exists())

    def test_missing_rf_model_is_reported_and_manual_retry_works(self):
        self.load()
        self.select_method("imbp_slic")
        with patch.object(imbp.config, "MODEL_PATH", self.directory / "missing.joblib"):
            self.window.segment_image()
            self.wait_for(lambda: self.window._worker is None)
            self.window.show_error.assert_called_once()
            self.assertIn("RF-модель", self.window.show_error.call_args.args[1])
            self.assertIsNone(self.window.result)
            self.assertTrue(self.window.btn_segment.isEnabled())
            self.window.show_error.reset_mock()
            self.select_mode("manual")
            self.process()

    def test_small_window_keeps_new_panels_and_status_usable(self):
        self.load()
        self.window.resize(900, 600)
        for method in ("imbp_otsu", "imbp_slic"):
            self.select_method(method)
            self.select_mode("analytical")
            self.process()
            self.app.processEvents()
            scroll = self.window.findChild(QtWidgets.QScrollArea)
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            details = self.window.lbl_details
            bottom = details.mapTo(self.window, QtCore.QPoint(0, details.height())).y()
            self.assertLessEqual(bottom, self.window.height())
            self.assertGreater(scroll.viewport().height(), 100)

    def test_loading_another_image_clears_mask_details_and_selection(self):
        self.load()
        self.select_method("imbp_otsu")
        self.process()
        self.window.rb_mask.setChecked(True)
        self.load()
        self.assertIsNone(self.window.mask_image)
        self.assertEqual(self.window.lbl_details.text(), "")
        self.assertTrue(self.window.rb_orig.isChecked())
        self.assertFalse(self.window.rb_mask.isEnabled())


if __name__ == "__main__":
    unittest.main()
