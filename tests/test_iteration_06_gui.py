import json
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

import test_iteration_01_gui as previous
from Bacteria_finder_pipeline import imbp_2_resolution as resolution


class ResolutionGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous.GuiIntegrationTests.setUpClass.__func__(cls)
        source = cv2.imread(str(previous.ROOT / "test_input_1.bmp"))
        cls.image = source[580:677, 990:1119].copy()

    setUp = previous.GuiIntegrationTests.setUp
    tearDown = previous.GuiIntegrationTests.tearDown
    wait_for = previous.GuiIntegrationTests.wait_for
    load = previous.GuiIntegrationTests.load
    process = previous.GuiIntegrationTests.process

    def select(self, stage, value):
        combo = self.window.stage_combos[stage]
        index = combo.findData(value)
        self.assertNotEqual(index, -1)
        combo.setCurrentIndex(index)
        self.app.processEvents()

    def enable_resolution(self, analysis=False):
        self.select("segmentation", "imbp_otsu")
        self.select("instances", "imbp_watershed")
        self.select("preprocessing", "imbp_2_resolution")
        if analysis:
            self.select("morphology", "imbp_ellipse")
            self.select("gram_classification", "imbp_od")

    def test_choice_label_defaults_and_selection_memory_for_every_segmenter(self):
        self.assertFalse(self.window.lbl_preprocessing_note.isVisible())
        self.assertFalse(self.window.rb_processed.isEnabled())
        for method in ("light_ridges", "imbp_otsu", "imbp_slic", "imbp_gmm", "imbp_auto"):
            self.select("segmentation", method)
            default = "light_average_pool" if method == "light_ridges" else "none"
            self.assertEqual(self.window.current_settings().preprocessing, default)
            self.select("preprocessing", "imbp_2_resolution")
            self.assertEqual(self.window.stage_combos["preprocessing"].currentText(), "Улучшение ×2 (imbp_2)")
            self.assertTrue(self.window.lbl_preprocessing_note.isVisible())
        self.select("segmentation", "light_ridges")
        self.assertEqual(self.window.current_settings().preprocessing, "imbp_2_resolution")
        self.select("preprocessing", "none")
        self.assertFalse(self.window.lbl_preprocessing_note.isVisible())

    def test_real_enhancement_runs_in_worker_with_live_event_loop(self):
        self.load()
        self.enable_resolution()
        ticks = []
        timer = QtCore.QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start()
        self.window.segment_image()
        worker = self.window._worker
        for control in (self.window.stage_combos["preprocessing"], self.window.btn_segment, self.window.btn_save, self.window.rb_processed):
            self.assertFalse(control.isEnabled())
        self.window.segment_image()
        self.assertIs(self.window._worker, worker)
        self.wait_for(lambda: self.window._worker is None)
        timer.stop()
        self.window.show_error.assert_not_called()
        self.assertGreater(len(ticks), 0)
        self.assertEqual(self.window.result_thread, self.app.thread())
        self.assertEqual(self.window.result.processed_shape, (194, 258))
        self.assertIn("129×97", self.window.lbl_dimensions.text())
        self.assertIn("258×194", self.window.lbl_dimensions.text())
        self.assertTrue(self.window.rb_processed.isEnabled())
        self.assertFalse(self.window.progress.isVisible())

    def test_every_view_saves_its_own_pixels_and_dimensions_to_unicode_path(self):
        self.load()
        self.enable_resolution(analysis=True)
        self.process()
        result = self.window.result
        np.testing.assert_array_equal(self.window.original_image, self.image)
        cases = (
            (self.window.rb_orig, self.image, "оригинал"),
            (self.window.rb_processed, result.processed_image_bgr, "предобработка"),
            (self.window.rb_seg, result.overlay_image_bgr, "сегмент"),
            (self.window.rb_mask, self.window.mask_image, "маска"),
            (self.window.rb_morphology, result.morphology_image_bgr, "форма"),
            (self.window.rb_gram, result.gram_image_bgr, "окраска"),
        )
        for radio, expected, name in cases:
            with self.subTest(view=name):
                radio.setChecked(True)
                np.testing.assert_array_equal(self.window.current_image(), expected)
                target = self.directory / (name + ".png")
                with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "PNG (*.png)")):
                    self.window.save_image()
                saved = cv2.imdecode(np.fromfile(target, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(saved, expected)
                self.assertEqual(saved.shape[:2], (97, 129) if radio is self.window.rb_orig else (194, 258))

    def test_json_and_object_table_identify_working_coordinates(self):
        self.load()
        self.enable_resolution(analysis=True)
        self.process()
        self.assertGreater(self.window.result.object_count, 0)
        target = self.directory / "анализ"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "JSON (*.json)")):
            self.window.save_classification()
        data = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(data["image_shape"], [194, 258])
        self.assertEqual(data["source_image_shape"], [97, 129])
        self.assertEqual(data["coordinate_system"]["scale_xy_from_source"], [2.0, 2.0])
        self.assertEqual(data["settings"]["preprocessing"], "imbp_2_resolution")

        def inspect(dialog):
            labels = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
            self.assertTrue(any("рабочего изображения 258×194" in label for label in labels))
            table = dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table.rowCount(), self.window.result.object_count)
            for row, obj in enumerate(self.window.result.classification.objects):
                self.assertEqual(table.item(row, 0).text(), obj.object_id)
                self.assertEqual(table.item(row, 1).text(), str(obj.area))
            return 0

        with patch.object(QtWidgets.QDialog, "exec_", inspect):
            self.window.show_classification_details()

    def test_switching_preprocessing_and_loading_image_clear_enlarged_results(self):
        self.load()
        self.enable_resolution()
        self.process()
        self.window.rb_processed.setChecked(True)
        self.select("preprocessing", "none")
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.rb_processed.isEnabled())
        self.assertTrue(self.window.rb_orig.isChecked())
        self.assertNotIn("рабочее", self.window.lbl_dimensions.text())
        self.process()
        self.assertEqual(self.window.result.processed_shape, self.image.shape[:2])
        self.select("preprocessing", "imbp_2_resolution")
        self.process()
        self.load()
        self.assertIsNone(self.window.result)
        self.assertTrue(self.window.rb_orig.isChecked())
        self.assertFalse(self.window.rb_processed.isEnabled())

    def test_missing_vendor_source_reports_error_and_retry_succeeds(self):
        self.load()
        self.enable_resolution()
        resolution._load_functions.cache_clear()
        try:
            with patch.object(resolution, "SOURCE", self.directory / "missing.py"):
                self.window.segment_image()
                self.wait_for(lambda: self.window._worker is None)
        finally:
            resolution._load_functions.cache_clear()
        self.window.show_error.assert_called_once()
        self.assertIn("Восстановите", self.window.show_error.call_args.args[1])
        self.assertIsNone(self.window.result)
        self.assertTrue(self.window.btn_segment.isEnabled())
        self.assertFalse(self.window.rb_processed.isEnabled())
        self.window.show_error.reset_mock()
        self.process()

    def test_closing_during_native_channel_finishes_worker_without_result(self):
        self.load()
        self.enable_resolution()
        started = threading.Event()
        native = resolution._ImageStore.open

        def delayed(store, path):
            image = native(store, path)
            if path.endswith("_channel0_ORIG.jpg"):
                started.set()
                time.sleep(0.15)
            return image

        with patch.object(resolution._ImageStore, "open", delayed):
            self.window.segment_image()
            self.wait_for(started.is_set)
            self.window.close()
            self.assertTrue(self.window.isVisible())
            self.wait_for(lambda: self.window._worker is None)
        self.assertFalse(self.window.isVisible())
        self.assertIsNone(self.window.result)
        self.window.show_error.assert_not_called()

    def test_small_window_fits_method_label_views_and_status(self):
        self.load()
        self.enable_resolution(analysis=True)
        self.process()
        for width, height in ((900, 600), (1200, 800)):
            self.window.resize(width, height)
            self.app.processEvents()
            scroll = self.window.findChild(QtWidgets.QScrollArea)
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            combo = self.window.stage_combos["preprocessing"]
            option = QtWidgets.QStyleOptionComboBox()
            combo.initStyleOption(option)
            field = combo.style().subControlRect(QtWidgets.QStyle.CC_ComboBox, option, QtWidgets.QStyle.SC_ComboBoxEditField, combo)
            self.assertLessEqual(combo.fontMetrics().horizontalAdvance(combo.currentText()), field.width())
            status = self.window.lbl_dimensions
            self.assertLessEqual(status.mapTo(self.window, QtCore.QPoint(0, status.height())).y(), self.window.height())
            self.assertGreater(scroll.viewport().height(), 100)


if __name__ == "__main__":
    unittest.main()
