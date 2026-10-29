import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

import Bacteria_finder_pipeline.worker as worker_module
from Bacteria_finder_pipeline import PipelineSettings, run_pipeline


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "iteration_01_gui", ROOT / "Bacteria_finder_GUI/Bacteria_finder_light_v.0.4_main_window.py"
)
GUI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUI)


class ObservedWindow(GUI.MainWindow):
    @QtCore.pyqtSlot(object)
    def on_result(self, result):
        self.result_thread = QtCore.QThread.currentThread()
        super().on_result(result)


class GuiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)
        image = cv2.imdecode(
            np.fromfile(ROOT / "test_input_1.bmp", dtype=np.uint8), cv2.IMREAD_COLOR
        )
        cls.image = cv2.resize(image, (384, 256))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / "проверка"
        self.directory.mkdir()
        self.source = self.directory / "вход.png"
        cv2.imencode(".png", self.image)[1].tofile(self.source)
        self.window = ObservedWindow()
        self.window.show_error = Mock()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        if self.window._worker is not None:
            self.window._worker.requestInterruption()
            self.wait_for(lambda: self.window._worker is None)
        self.window.close()
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()

    def wait_for(self, condition, timeout=30000):
        if condition():
            return
        loop = QtCore.QEventLoop()
        poll = QtCore.QTimer()
        poll.setInterval(5)
        poll.timeout.connect(lambda: loop.quit() if condition() else None)
        deadline = QtCore.QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start()
        deadline.start(timeout)
        loop.exec_()
        poll.stop()
        deadline.stop()
        self.assertTrue(condition(), "Истекло время ожидания фоновой обработки")

    def load(self, filename=None):
        with patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(filename or self.source), "")):
            self.window.load_image()

    def process(self):
        self.window.segment_image()
        self.wait_for(lambda: self.window._worker is None)
        self.assertIsNotNone(self.window.result)
        self.window.show_error.assert_not_called()

    def delayed_pipeline(self, *args, **kwargs):
        self.compute_thread = QtCore.QThread.currentThread()
        time.sleep(0.15)
        return run_pipeline(*args, **kwargs)

    def test_sidebar_has_no_horizontal_scroll_and_status_remains_visible(self):
        for size in ((1200, 800), (900, 600)):
            with self.subTest(size=size):
                self.window.resize(*size)
                self.app.processEvents()
                scroll = self.window.findChild(QtWidgets.QScrollArea)
                self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                status = self.window.lbl_status
                bottom = status.mapTo(self.window, QtCore.QPoint(0, status.height())).y()
                self.assertLessEqual(bottom, self.window.height())

    def test_background_processing_keeps_event_loop_live_and_prevents_duplicate_run(self):
        self.load()
        ticks = []
        timer = QtCore.QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: ticks.append(1))
        with patch.object(worker_module, "run_pipeline", side_effect=self.delayed_pipeline):
            timer.start()
            self.window.segment_image()
            first_worker = self.window._worker
            self.assertFalse(self.window.btn_segment.isEnabled())
            self.assertFalse(self.window.btn_load.isEnabled())
            self.assertFalse(self.window.spin_sigma.isEnabled())
            self.window.segment_image()
            self.assertIs(first_worker, self.window._worker)
            self.wait_for(lambda: self.window._worker is None)
            timer.stop()
        self.assertGreaterEqual(len(ticks), 3)
        self.assertNotEqual(self.compute_thread, self.app.thread())
        self.assertEqual(self.window.result_thread, self.app.thread())
        self.assertIsNotNone(self.window.result)
        self.assertTrue(self.window.rb_seg.isChecked())
        self.assertTrue(self.window.btn_segment.isEnabled())
        self.assertFalse(self.window.progress.isVisible())

    def test_settings_invalidate_result_and_rerun_uses_new_sigma(self):
        self.load()
        self.process()
        self.window.spin_sigma.setValue(4.0)
        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window.overlay_image)
        self.assertEqual(self.window.lbl_count.text(), "Объекты: 0")
        self.assertEqual(self.window.lbl_time.text(), "Время: —")
        self.assertTrue(self.window.rb_orig.isChecked())
        self.assertFalse(self.window.rb_seg.isEnabled())
        self.process()
        expected = run_pipeline(self.image, PipelineSettings(sigma=4.0))
        np.testing.assert_array_equal(self.window.overlay_image, expected.overlay_image_bgr)

    def test_new_image_clears_result_and_bad_file_keeps_previous_image(self):
        self.load()
        self.process()
        old_result = self.window.result
        for content in (b"", b"not an image"):
            bad = self.directory / "ошибка.png"
            bad.write_bytes(content)
            self.load(bad)
            self.assertIs(self.window.result, old_result)
            np.testing.assert_array_equal(self.window.original_image, self.image)
        self.assertEqual(self.window.show_error.call_count, 2)
        other = self.directory / "другое.bmp"
        blank = np.full((80, 120, 3), 255, np.uint8)
        cv2.imencode(".bmp", blank)[1].tofile(other)
        self.load(other)
        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window.overlay_image)
        np.testing.assert_array_equal(self.window.original_image, blank)

    def test_unicode_paths_save_current_view_and_add_missing_extension(self):
        self.load()
        self.process()
        for segmented in (True, False):
            with self.subTest(segmented=segmented):
                self.window.rb_seg.setChecked(segmented)
                self.window.rb_orig.setChecked(not segmented)
                target = self.directory / ("сегмент" if segmented else "оригинал")
                with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "PNG (*.png)")):
                    self.window.save_image()
                saved = cv2.imdecode(np.fromfile(target.with_suffix(".png"), dtype=np.uint8), cv2.IMREAD_COLOR)
                expected = self.window.overlay_image if segmented else self.image
                np.testing.assert_array_equal(saved, expected)
        self.window.show_error.assert_not_called()

    def test_save_failure_is_reported_without_losing_result(self):
        self.load()
        self.process()
        result = self.window.result
        for target in (self.directory / "bad.txt", self.directory / "missing" / "result.png"):
            with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "PNG (*.png)")):
                self.window.save_image()
            self.assertIs(self.window.result, result)
        self.assertEqual(self.window.show_error.call_count, 2)

    def test_worker_failure_restores_controls_and_allows_retry(self):
        self.load()
        with patch.object(worker_module, "run_pipeline", side_effect=RuntimeError("Проверочная ошибка")):
            self.window.segment_image()
            self.wait_for(lambda: self.window._worker is None)
        self.window.show_error.assert_called_once()
        self.assertIn("Проверочная ошибка", self.window.show_error.call_args.args[1])
        self.assertTrue(self.window.btn_segment.isEnabled())
        self.assertTrue(self.window.btn_load.isEnabled())
        self.assertIsNone(self.window.result)
        self.window.show_error.reset_mock()
        self.process()

    def test_close_during_processing_waits_without_destroying_active_thread(self):
        self.load()
        with patch.object(worker_module, "run_pipeline", side_effect=self.delayed_pipeline):
            self.window.segment_image()
            self.window.close()
            self.assertTrue(self.window._close_requested)
            self.assertTrue(self.window.isVisible())
            self.wait_for(lambda: self.window._worker is None)
        self.assertFalse(self.window.isVisible())
        self.assertIsNone(self.window.result)
        self.window.show_error.assert_not_called()

    def test_result_from_obsolete_settings_is_not_applied(self):
        self.load()
        with patch.object(worker_module, "run_pipeline", side_effect=self.delayed_pipeline):
            self.window.segment_image()
            self.window.spin_sigma.setValue(4.0)
            self.wait_for(lambda: self.window._worker is None)
        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window.overlay_image)
        self.assertTrue(self.window.rb_orig.isChecked())


if __name__ == "__main__":
    unittest.main()
