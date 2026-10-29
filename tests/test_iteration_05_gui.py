import json
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

import test_iteration_01_gui as previous
from third_party.imbp.src import imbp


class ClassificationGuiTests(unittest.TestCase):
    setUpClass = classmethod(previous.GuiIntegrationTests.setUpClass.__func__)
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

    def enable_analysis(self):
        self.select("morphology", "imbp_ellipse")
        self.select("gram_classification", "imbp_od")

    def test_independent_choices_are_available_and_remembered_for_all_segmenters(self):
        self.assertFalse(self.window.gb_classification.isVisible())
        self.assertFalse(self.window.rb_morphology.isVisible())
        self.assertFalse(self.window.rb_gram.isVisible())
        for method in ("light_ridges", "imbp_otsu", "imbp_slic", "imbp_gmm", "imbp_auto"):
            self.select("segmentation", method)
            self.assertEqual(self.window.current_settings().morphology, "none")
            self.assertEqual(self.window.current_settings().gram_classification, "none")
            self.enable_analysis()
            self.assertTrue(self.window.gb_classification.isVisible())
            self.assertTrue(self.window.spin_eccentricity.isEnabled())
            self.assertFalse(self.window.spin_red_delta.isEnabled())
        self.select("segmentation", "light_ridges")
        self.assertEqual(self.window.current_settings().morphology, "imbp_ellipse")
        self.assertEqual(self.window.current_settings().gram_classification, "imbp_od")

    def test_worker_independent_outputs_and_selected_preview(self):
        self.load()
        self.select("segmentation", "imbp_otsu")
        self.select("instances", "imbp_watershed")
        for morph, gram in (("imbp_ellipse", "none"), ("none", "imbp_od"), ("none", "imbp_rgb"), ("imbp_ellipse", "imbp_od")):
            with self.subTest(morph=morph, gram=gram):
                self.select("morphology", morph)
                self.select("gram_classification", gram)
                self.process()
                self.assertIsNotNone(self.window.result.classification)
                self.assertEqual(self.window.rb_morphology.isEnabled(), morph != "none")
                self.assertEqual(self.window.rb_gram.isEnabled(), gram != "none")
                self.assertEqual(self.window.rb_morphology.isChecked(), morph != "none")
                self.assertEqual(self.window.rb_gram.isChecked(), morph == "none")
                self.assertTrue(self.window.btn_classification_details.isEnabled())
                self.assertTrue(self.window.btn_export_classification.isEnabled())
                if morph == "none":
                    self.assertNotIn("Кокки", self.window.lbl_classification.text())
                if gram == "none":
                    self.assertNotIn("Грам+", self.window.lbl_classification.text())
        self.assertIs(self.window.result_thread, self.app.thread())

    def test_parameter_changes_reset_result_and_controls_are_locked_while_busy(self):
        self.load()
        self.enable_analysis()
        self.process()
        self.window.spin_eccentricity.setValue(0.8)
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.btn_export_classification.isEnabled())
        self.assertEqual(self.window.lbl_classification.text(), "")
        self.assertTrue(self.window.rb_orig.isChecked())
        self.select("gram_classification", "imbp_rgb")
        self.assertTrue(self.window.spin_red_delta.isEnabled())
        self.window.spin_red_delta.setValue(30)
        self.window.spin_noise_area.setValue(20)
        self.window.segment_image()
        for control in (self.window.spin_noise_area, self.window.spin_eccentricity, self.window.spin_red_delta, self.window.btn_classification_details, self.window.btn_export_classification):
            self.assertFalse(control.isEnabled())
        self.wait_for(lambda: self.window._worker is None)
        self.window.show_error.assert_not_called()
        params = self.window.result.settings.classification_params
        self.assertEqual((params.noise_min_area, params.ecc_threshold, params.red_delta), (20, 0.8, 30))

    def test_analysis_image_and_json_export_with_unicode_paths(self):
        self.load()
        self.enable_analysis()
        self.process()
        result = self.window.result
        for radio, expected, name in ((self.window.rb_morphology, result.morphology_image_bgr, "форма.png"), (self.window.rb_gram, result.gram_image_bgr, "окраска.png")):
            radio.setChecked(True)
            target = self.directory / name
            with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "PNG (*.png)")):
                self.window.save_image()
            np.testing.assert_array_equal(cv2.imdecode(np.fromfile(target, dtype=np.uint8), cv2.IMREAD_COLOR), expected)
        target = self.directory / "анализ"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "JSON (*.json)")):
            self.window.save_classification()
        content = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(content["summary"], result.classification.summary)
        self.assertEqual(len(content["objects"]), result.object_count)
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(self.directory / "bad.txt"), "JSON (*.json)")):
            self.window.save_classification()
        self.window.show_error.assert_called_once()
        self.assertFalse((self.directory / "bad.txt").exists())

    def test_object_table_ids_values_and_disabled_morphotype(self):
        self.load()
        self.select("gram_classification", "imbp_od")
        self.process()

        def inspect(dialog):
            table = dialog.findChild(QtWidgets.QTableWidget)
            objects = self.window.result.classification.objects
            self.assertEqual(table.rowCount(), len(objects))
            self.assertEqual(table.columnCount(), 12)
            for row, obj in enumerate(objects):
                self.assertEqual(table.item(row, 0).text(), obj.object_id)
                self.assertEqual(table.item(row, 1).text(), str(obj.area))
                self.assertEqual(table.item(row, 2).text(), "Да" if obj.artifact else "Нет")
                self.assertEqual(table.item(row, 3).text(), "—")
            return 0

        with patch.object(QtWidgets.QDialog, "exec_", inspect):
            self.window.show_classification_details()

    def test_small_window_can_reach_analysis_panel_and_status_is_visible(self):
        self.load()
        self.window.resize(900, 600)
        self.select("segmentation", "imbp_auto")
        self.select("filtering", "imbp_rf")
        self.select("instances", "imbp_watershed")
        self.enable_analysis()
        self.process()
        scroll = self.window.findChild(QtWidgets.QScrollArea)
        self.app.processEvents()
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
        self.assertGreater(scroll.viewport().height(), 100)
        for target in (self.window.spin_noise_area, self.window.btn_classification_details, self.window.btn_export_classification):
            scroll.ensureWidgetVisible(target)
            self.app.processEvents()
            point = target.mapTo(scroll.viewport(), QtCore.QPoint(0, 0))
            self.assertGreaterEqual(point.y(), 0)
            self.assertLessEqual(point.y() + target.height(), scroll.viewport().height())
        for stage in ("morphology", "gram_classification"):
            combo = self.window.stage_combos[stage]
            option = QtWidgets.QStyleOptionComboBox()
            combo.initStyleOption(option)
            field = combo.style().subControlRect(QtWidgets.QStyle.CC_ComboBox, option, QtWidgets.QStyle.SC_ComboBoxEditField, combo)
            self.assertLessEqual(combo.fontMetrics().horizontalAdvance(combo.currentText()), field.width())
        label = self.window.lbl_details
        self.assertLessEqual(label.mapTo(self.window, QtCore.QPoint(0, label.height())).y(), self.window.height())

    def test_classification_failure_recovers_for_retry(self):
        self.load()
        self.enable_analysis()
        with patch.object(imbp, "classify_objects", side_effect=RuntimeError("Ошибка классификации")):
            self.window.segment_image()
            self.wait_for(lambda: self.window._worker is None)
        self.window.show_error.assert_called_once()
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.btn_export_classification.isEnabled())
        self.assertTrue(self.window.btn_segment.isEnabled())
        self.window.show_error.reset_mock()
        self.process()

    def test_close_during_classification_waits_for_native_function(self):
        self.load()
        self.enable_analysis()
        started = threading.Event()
        native = imbp.classify_objects

        def delayed(*args, **kwargs):
            started.set()
            time.sleep(0.15)
            return native(*args, **kwargs)

        with patch.object(imbp, "classify_objects", side_effect=delayed):
            self.window.segment_image()
            self.wait_for(started.is_set)
            self.window.close()
            self.assertTrue(self.window.isVisible())
            self.wait_for(lambda: self.window._worker is None)
        self.assertFalse(self.window.isVisible())
        self.assertIsNone(self.window.result)
        self.window.show_error.assert_not_called()

    def test_loading_and_disabling_analysis_clear_all_derived_views(self):
        self.load()
        self.enable_analysis()
        self.process()
        self.window.rb_gram.setChecked(True)
        self.load()
        self.assertIsNone(self.window.result)
        self.assertTrue(self.window.rb_orig.isChecked())
        self.assertFalse(self.window.rb_gram.isEnabled())
        self.assertFalse(self.window.btn_classification_details.isEnabled())
        self.assertEqual(self.window.lbl_classification.text(), "")
        self.select("morphology", "none")
        self.select("gram_classification", "none")
        self.process()
        self.assertIsNone(self.window.result.classification)
        self.assertFalse(self.window.gb_classification.isVisible())
        self.assertFalse(self.window.rb_morphology.isVisible())
        self.assertFalse(self.window.rb_gram.isVisible())
        self.assertTrue(self.window.rb_seg.isChecked())


if __name__ == "__main__":
    unittest.main()
