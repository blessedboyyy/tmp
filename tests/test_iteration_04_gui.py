import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

import test_iteration_01_gui as previous
from third_party.imbp.src import imbp


class PostprocessingGuiTests(unittest.TestCase):
    setUpClass = classmethod(previous.GuiIntegrationTests.setUpClass.__func__)
    setUp = previous.GuiIntegrationTests.setUp
    tearDown = previous.GuiIntegrationTests.tearDown
    wait_for = previous.GuiIntegrationTests.wait_for
    load = previous.GuiIntegrationTests.load
    process = previous.GuiIntegrationTests.process

    def select(self, stage, method):
        combo = self.window.stage_combos[stage]
        index = combo.findData(method)
        self.assertNotEqual(index, -1)
        combo.setCurrentIndex(index)
        self.app.processEvents()

    def test_stage_choices_are_independent_and_remembered_per_method(self):
        self.assertEqual(self.window.stage_combos["filtering"].findData("imbp_rf"), -1)
        self.assertEqual(self.window.stage_combos["instances"].findData("imbp_watershed"), -1)
        self.select("segmentation", "imbp_otsu")
        self.assertEqual(self.window.current_settings().filtering, "imbp_native")
        self.select("filtering", "imbp_rf")
        self.assertTrue(self.window.gb_filter.isVisible())
        self.assertFalse(self.window.gb_instances.isVisible())
        self.assertIsNone(self.window.current_settings().object_filter.threshold)
        self.assertFalse(self.window.spin_filter_threshold.isEnabled())
        self.select("instances", "imbp_watershed")
        self.assertTrue(self.window.gb_instances.isVisible())
        self.select("segmentation", "imbp_gmm")
        self.assertEqual(self.window.current_settings().filtering, "imbp_native")
        self.assertEqual(self.window.current_settings().instances, "connected_components")
        self.select("segmentation", "imbp_otsu")
        self.assertEqual(self.window.current_settings().filtering, "imbp_rf")
        self.assertEqual(self.window.current_settings().instances, "imbp_watershed")

    def test_filter_only_split_only_and_combined_counts_and_resets(self):
        self.load()
        self.select("segmentation", "imbp_otsu")
        for filtering, splitting in (("imbp_rf", "connected_components"), ("imbp_native", "imbp_watershed"), ("imbp_rf", "imbp_watershed")):
            with self.subTest(filtering=filtering, splitting=splitting):
                self.select("filtering", filtering)
                self.select("instances", splitting)
                self.process()
                label = "Клетки" if splitting == "imbp_watershed" else "Компоненты маски"
                self.assertTrue(self.window.lbl_count.text().startswith(label + ":"))
                self.assertEqual(self.window.result.segmentation.filter_probabilities is not None, filtering == "imbp_rf")
        self.window.rb_mask.setChecked(True)
        self.window.instance_spins["h"].setValue(2)
        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window.mask_image)
        self.assertEqual(self.window.lbl_count.text(), "Клетки: 0")
        self.assertTrue(self.window.rb_orig.isChecked())
        self.process()
        self.select("instances", "connected_components")
        self.assertEqual(self.window.lbl_count.text(), "Компоненты маски: 0")

    def test_manual_threshold_and_busy_controls(self):
        self.load()
        self.select("segmentation", "imbp_otsu")
        self.select("filtering", "imbp_rf")
        self.select("instances", "imbp_watershed")
        self.window.check_filter_model_threshold.setChecked(False)
        self.assertTrue(self.window.spin_filter_threshold.isEnabled())
        self.window.spin_filter_threshold.setValue(0.5)
        self.window.segment_image()
        self.assertFalse(self.window.spin_filter_threshold.isEnabled())
        self.assertFalse(self.window.check_filter_model_threshold.isEnabled())
        self.assertFalse(self.window.instance_spins["h"].isEnabled())
        self.wait_for(lambda: self.window._worker is None)
        self.window.show_error.assert_not_called()
        self.assertEqual(self.window.result.segmentation.filter_threshold, 0.5)
        self.window.check_filter_model_threshold.setChecked(True)
        self.assertIsNone(self.window.result)
        self.process()
        self.assertEqual(self.window.result.segmentation.filter_threshold, 0.1)

    def test_auto_keeps_tile_table_after_filter_and_split(self):
        self.load()
        self.select("segmentation", "imbp_auto")
        self.select("filtering", "imbp_rf")
        self.select("instances", "imbp_watershed")
        self.process()
        self.assertTrue(self.window.btn_auto_details.isEnabled())
        self.assertTrue(self.window.result.segmentation.tile_methods)
        self.assertIn("Авто:", self.window.lbl_details.text())
        self.assertIn("RF:", self.window.lbl_details.text())
        self.assertIn("Водораздел IMBP:", self.window.lbl_details.text())

    def test_missing_filter_restores_controls_and_allows_unfiltered_retry(self):
        self.load()
        self.select("segmentation", "imbp_otsu")
        self.select("filtering", "imbp_rf")
        self.select("instances", "imbp_watershed")
        with patch.object(imbp.config, "OBJECT_FILTER_PATH", self.directory / "missing.joblib"):
            self.window.segment_image()
            self.wait_for(lambda: self.window._worker is None)
            self.window.show_error.assert_called_once()
            self.assertIn("RF-фильтр недоступен", self.window.show_error.call_args.args[1])
            self.assertIsNone(self.window.result)
            self.assertTrue(self.window.btn_segment.isEnabled())
            self.window.show_error.reset_mock()
            self.select("filtering", "imbp_native")
            self.process()
            self.assertIsNone(self.window.result.segmentation.filter_probabilities)

    def test_exports_binary_mask_and_overlay_with_cell_boundaries(self):
        self.load()
        self.select("segmentation", "imbp_otsu")
        self.select("filtering", "imbp_rf")
        self.select("instances", "imbp_watershed")
        self.process()
        for radio, expected, name in (
            (self.window.rb_mask, self.window.mask_image, "маска.png"),
            (self.window.rb_seg, self.window.overlay_image, "клетки.png"),
        ):
            radio.setChecked(True)
            target = self.directory / name
            with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "PNG (*.png)")):
                self.window.save_image()
            decoded = cv2.imdecode(np.fromfile(target, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, expected)

    def test_small_window_keeps_both_new_panels_accessible(self):
        self.load()
        self.window.resize(900, 600)
        self.select("segmentation", "imbp_otsu")
        self.select("filtering", "imbp_rf")
        self.select("instances", "imbp_watershed")
        self.process()
        self.app.processEvents()
        scroll = self.window.findChild(QtWidgets.QScrollArea)
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
        self.assertGreater(scroll.viewport().height(), 100)
        combo = self.window.stage_combos["filtering"]
        option = QtWidgets.QStyleOptionComboBox()
        combo.initStyleOption(option)
        field = combo.style().subControlRect(QtWidgets.QStyle.CC_ComboBox, option, QtWidgets.QStyle.SC_ComboBoxEditField, combo)
        self.assertLessEqual(combo.fontMetrics().horizontalAdvance(combo.currentText()), field.width())
        for panel in (self.window.gb_filter, self.window.gb_instances):
            scroll.ensureWidgetVisible(panel)
            self.app.processEvents()
            self.assertTrue(panel.isVisible())
        label = self.window.lbl_details
        bottom = label.mapTo(self.window, QtCore.QPoint(0, label.height())).y()
        self.assertLessEqual(bottom, self.window.height())

    def test_close_during_splitting_waits_for_native_function(self):
        self.load()
        self.select("segmentation", "imbp_otsu")
        self.select("instances", "imbp_watershed")
        started = threading.Event()
        native = imbp.instances.split_instances

        def delayed(*args, **kwargs):
            started.set()
            time.sleep(0.15)
            return native(*args, **kwargs)

        with patch.object(imbp.instances, "split_instances", side_effect=delayed):
            self.window.segment_image()
            self.wait_for(started.is_set)
            self.window.close()
            self.assertTrue(self.window.isVisible())
            self.wait_for(lambda: self.window._worker is None)
        self.assertFalse(self.window.isVisible())
        self.assertIsNone(self.window.result)
        self.window.show_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
