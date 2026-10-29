import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

import test_iteration_01_gui as previous
from Bacteria_finder_pipeline import GmmSettings, PipelineSettings, run_pipeline
from third_party.imbp.src import imbp
from third_party.imbp.src.imbp.segmentation import fusion


class GmmAutoGuiTests(unittest.TestCase):
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

    def test_new_method_panels_restore_settings_and_compatible_stages(self):
        self.load()
        self.select_method("imbp_gmm")
        self.assertTrue(self.window.gb_gmm.isVisible())
        self.assertFalse(self.window.gb_auto.isVisible())
        self.assertEqual(self.window.current_settings().gmm, GmmSettings())
        self.window.spin_gmm_components.setValue(4)
        prep = self.window.stage_combos["preprocessing"]
        prep.setCurrentIndex(prep.findData("light_average_pool"))
        self.select_method("imbp_auto")
        self.assertTrue(self.window.gb_auto.isVisible())
        self.assertFalse(self.window.gb_gmm.isVisible())
        self.assertFalse(self.window.btn_auto_details.isEnabled())
        settings = self.window.current_settings()
        self.assertEqual((settings.preprocessing, settings.filtering, settings.instances), ("none", "imbp_native", "connected_components"))
        self.select_method("imbp_gmm")
        self.assertEqual(self.window.current_settings().gmm.n_components, 4)
        self.assertEqual(self.window.current_settings().preprocessing, "light_average_pool")
        self.select_method("light_ridges")
        self.assertFalse(self.window.gb_gmm.isVisible())
        self.assertFalse(self.window.gb_auto.isVisible())
        self.assertEqual(self.window.current_settings().sigma, 2.5)

    def test_both_new_methods_run_in_worker_and_restore_controls(self):
        self.load()
        for method in ("imbp_gmm", "imbp_auto"):
            with self.subTest(method=method):
                self.select_method(method)
                self.window.segment_image()
                first = self.window._worker
                self.window.segment_image()
                self.assertIs(first, self.window._worker)
                self.assertFalse(self.window.spin_gmm_components.isEnabled())
                self.assertFalse(self.window.btn_auto_details.isEnabled())
                self.assertFalse(self.window.btn_load.isEnabled())
                self.wait_for(lambda: self.window._worker is None)
                self.window.show_error.assert_not_called()
                self.assertIsNotNone(self.window.result)
                self.assertEqual(self.window.result_thread, self.app.thread())
                self.assertTrue(self.window.btn_segment.isEnabled())
                self.assertTrue(self.window.rb_mask.isEnabled())
                self.assertEqual(self.window.btn_auto_details.isEnabled(), method == "imbp_auto")

    def test_gmm_parameter_changes_clear_and_recompute_the_result(self):
        self.load()
        self.select_method("imbp_gmm")
        self.process()
        self.window.rb_mask.setChecked(True)
        self.window.spin_gmm_components.setValue(4)
        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window.mask_image)
        self.assertTrue(self.window.rb_orig.isChecked())
        self.window.spin_gmm_min_area.setValue(0)
        self.window.check_gmm_bilateral.setChecked(False)
        self.window.spin_gmm_beta.setValue(1.0)
        self.process()
        expected = run_pipeline(self.image, PipelineSettings(
            segmentation="imbp_gmm", gmm=GmmSettings(n_components=4, min_area=0, bilateral=False, hmrf_beta=1.0)
        ))
        np.testing.assert_array_equal(self.window.mask_image, expected.segmentation.mask.astype(np.uint8) * 255)

    def test_auto_table_shows_selected_methods_and_one_based_inclusive_bounds(self):
        self.load()
        self.window.original_image = cv2.resize(self.image, (809, 611))
        self.select_method("imbp_auto")
        self.process()
        methods = self.window.result.segmentation.tile_methods
        seen = []

        def inspect(dialog):
            table = dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table.rowCount(), len(methods))
            self.assertEqual(table.editTriggers(), QtWidgets.QAbstractItemView.NoEditTriggers)
            names = {"otsu": "Оцу", "slic": "SLIC", "gmm": "GMM"}
            for row, tile in enumerate(methods):
                y0, y1, x0, x1 = tile.bbox
                self.assertEqual([table.item(row, column).text() for column in range(4)], [str(row + 1), f"{x0 + 1}–{x1}", f"{y0 + 1}–{y1}", names[tile.method]])
            seen.append(True)
            return 0

        with patch.object(QtWidgets.QDialog, "exec_", inspect):
            self.window.show_auto_details()
        self.assertEqual(seen, [True])
        self.load()
        self.assertFalse(self.window.btn_auto_details.isEnabled())
        self.assertIsNone(self.window.result)

    def test_auto_is_independent_of_saved_manual_settings(self):
        self.load()
        self.select_method("imbp_auto")
        self.process()
        expected = self.window.mask_image.copy()
        expected_methods = self.window.result.segmentation.tile_methods
        self.select_method("imbp_gmm")
        self.window.spin_gmm_min_area.setValue(100000)
        self.window.spin_gmm_beta.setValue(50)
        self.select_method("imbp_slic")
        self.window.combo_slic_mode.setCurrentIndex(self.window.combo_slic_mode.findData("manual"))
        self.window.spin_slic_k.setValue(1)
        self.select_method("imbp_otsu")
        self.window.check_otsu_auto.setChecked(False)
        self.window.spin_otsu_threshold.setValue(255)
        self.select_method("imbp_auto")
        self.process()
        np.testing.assert_array_equal(self.window.mask_image, expected)
        self.assertEqual(self.window.result.segmentation.tile_methods, expected_methods)

    def test_missing_chooser_recovers_for_gmm_and_then_auto(self):
        self.load()
        self.select_method("imbp_auto")
        with patch.object(imbp.config, "METHOD_CHOOSER_PATH", self.directory / "missing.joblib"):
            self.window.segment_image()
            self.wait_for(lambda: self.window._worker is None)
            self.window.show_error.assert_called_once()
            self.assertIn("Модель выбора метода", self.window.show_error.call_args.args[1])
            self.assertIsNone(self.window.result)
            self.assertTrue(self.window.btn_segment.isEnabled())
            self.assertFalse(self.window.btn_auto_details.isEnabled())
            self.window.show_error.reset_mock()
            self.select_method("imbp_gmm")
            self.process()
        self.select_method("imbp_auto")
        self.process()

    def test_both_new_methods_export_the_selected_mask(self):
        self.load()
        for method in ("imbp_gmm", "imbp_auto"):
            self.select_method(method)
            self.process()
            self.window.rb_mask.setChecked(True)
            target = self.directory / f"маска_{method}.png"
            with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "PNG (*.png)")):
                self.window.save_image()
            decoded = cv2.imdecode(np.fromfile(target, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, self.window.mask_image)
        self.window.show_error.assert_not_called()

    def test_small_window_keeps_new_controls_and_status_usable(self):
        self.load()
        self.window.resize(900, 600)
        for method in ("imbp_gmm", "imbp_auto"):
            self.select_method(method)
            self.process()
            self.app.processEvents()
            scroll = self.window.findChild(QtWidgets.QScrollArea)
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            self.assertGreater(scroll.viewport().height(), 100)
            label = self.window.lbl_details
            bottom = label.mapTo(self.window, QtCore.QPoint(0, label.height())).y()
            self.assertLessEqual(bottom, self.window.height())

    def test_close_during_auto_waits_for_author_tile_workers(self):
        self.load()
        self.select_method("imbp_auto")
        started = threading.Event()
        native = fusion.candidate_masks

        def delayed(*args, **kwargs):
            started.set()
            time.sleep(0.15)
            return native(*args, **kwargs)

        with patch.object(fusion, "candidate_masks", side_effect=delayed):
            self.window.segment_image()
            self.wait_for(started.is_set)
            self.window.close()
            self.assertTrue(self.window._close_requested)
            self.assertTrue(self.window.isVisible())
            self.wait_for(lambda: self.window._worker is None)
        self.assertFalse(self.window.isVisible())
        self.assertIsNone(self.window.result)
        self.window.show_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
