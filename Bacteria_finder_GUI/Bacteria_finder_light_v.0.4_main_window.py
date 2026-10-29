import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from Bacteria_finder_pipeline import STAGES, PipelineSettings, ProcessingResult
from Bacteria_finder_pipeline.worker import ProcessingWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.overlay_image = None
        self.result = None
        self._worker = None
        self._revision = 0
        self._active_revision = None
        self._close_requested = False
        self.init_ui()

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle("Bacteria Finder — Light")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)
        self.setFont(QtGui.QFont("Times New Roman", 14))

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(central)
        gb_image = QtWidgets.QGroupBox("Изображение")
        image_layout = QtWidgets.QVBoxLayout(gb_image)
        self.lbl_image = QtWidgets.QLabel("Загрузите изображение")
        self.lbl_image.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_image.setMinimumSize(160, 120)
        self.lbl_image.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )
        image_layout.addWidget(self.lbl_image)
        layout.addWidget(gb_image, 3)

        controls = QtWidgets.QWidget()
        ctrl = QtWidgets.QVBoxLayout(controls)
        self.btn_load = QtWidgets.QPushButton("Загрузить изображение")
        self.btn_load.clicked.connect(self.load_image)
        ctrl.addWidget(self.btn_load)
        self.btn_segment = QtWidgets.QPushButton("Сегментировать")
        self.btn_segment.clicked.connect(self.segment_image)
        ctrl.addWidget(self.btn_segment)
        self.btn_save = QtWidgets.QPushButton("Сохранить изображение")
        self.btn_save.clicked.connect(self.save_image)
        ctrl.addWidget(self.btn_save)

        gb_view = QtWidgets.QGroupBox("Режим просмотра")
        view_layout = QtWidgets.QVBoxLayout(gb_view)
        self.rb_orig = QtWidgets.QRadioButton("Оригинал")
        self.rb_seg = QtWidgets.QRadioButton("Сегмент")
        self.rb_orig.setChecked(True)
        view_layout.addWidget(self.rb_orig)
        view_layout.addWidget(self.rb_seg)
        self.rb_orig.toggled.connect(self.update_view)
        self.rb_seg.toggled.connect(self.update_view)
        ctrl.addWidget(gb_view)

        gb_stages = QtWidgets.QGroupBox("Этапы обработки")
        gb_stages.setFont(QtGui.QFont("Times New Roman", 12))
        stages_layout = QtWidgets.QFormLayout(gb_stages)
        stages_layout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.stage_combos = {}
        defaults = PipelineSettings()
        for stage in STAGES:
            combo = QtWidgets.QComboBox()
            for method in stage.methods:
                combo.addItem(method.name, method.id)
            combo.setCurrentIndex(combo.findData(getattr(defaults, stage.key)))
            combo.setToolTip(stage.description)
            combo.currentIndexChanged.connect(self.settings_changed)
            self.stage_combos[stage.key] = combo
            stages_layout.addRow(f"{stage.name}:", combo)
        ctrl.addWidget(gb_stages)

        gb_params = QtWidgets.QGroupBox("Параметры Light")
        params_layout = QtWidgets.QHBoxLayout(gb_params)
        params_layout.addWidget(QtWidgets.QLabel("σ сглаживания:"))
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setRange(0.1, 20.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(defaults.sigma)
        self.spin_sigma.valueChanged.connect(self.settings_changed)
        params_layout.addWidget(self.spin_sigma)
        ctrl.addWidget(gb_params)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        gb_status = QtWidgets.QGroupBox("Состояние")
        status_layout = QtWidgets.QVBoxLayout(gb_status)
        self.lbl_count = QtWidgets.QLabel("Объекты: 0")
        self.lbl_time = QtWidgets.QLabel("Время: —")
        self.lbl_status = QtWidgets.QLabel("Загрузите изображение")
        self.lbl_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_count)
        status_layout.addWidget(self.lbl_time)
        status_layout.addWidget(self.lbl_status)
        ctrl.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMinimumWidth(max(
            440,
            controls.minimumSizeHint().width()
            + scroll.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent) + 2,
        ))
        scroll.setWidget(controls)
        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(scroll, 1)
        sidebar_layout.addWidget(self.progress)
        sidebar_layout.addWidget(gb_status)
        layout.addWidget(sidebar, 1)
        self.setCentralWidget(central)
        self.update_controls()

    def current_settings(self) -> PipelineSettings:
        return PipelineSettings(
            **{key: combo.currentData() for key, combo in self.stage_combos.items()},
            sigma=self.spin_sigma.value(),
        )

    def update_controls(self):
        busy = self._worker is not None or self._close_requested
        has_image = self.original_image is not None
        self.btn_load.setEnabled(not busy)
        self.btn_segment.setEnabled(has_image and not busy)
        self.btn_save.setEnabled(has_image and not busy)
        self.spin_sigma.setEnabled(not busy)
        for combo in self.stage_combos.values():
            combo.setEnabled(not busy)
        self.rb_seg.setEnabled(self.result is not None and not busy)
        self.progress.setVisible(self._worker is not None)

    def clear_result(self):
        self.result = None
        self.overlay_image = None
        self.lbl_count.setText("Объекты: 0")
        self.lbl_time.setText("Время: —")
        self.rb_orig.setChecked(True)
        self.update_controls()
        self.update_view()

    def settings_changed(self):
        self._revision += 1
        self.clear_result()
        if self.original_image is not None:
            self.lbl_status.setText("Настройки изменены. Выполните сегментацию.")

    def show_error(self, title, message, details=""):
        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Critical)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        if details:
            dialog.setDetailedText(details)
        dialog.exec_()

    def load_image(self):
        if self._worker is not None or self._close_requested:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Открыть файл", "", "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if not filename:
            return
        try:
            data = np.fromfile(filename, dtype=np.uint8)
            if not data.size:
                raise ValueError("Выбранный файл пуст.")
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Не удалось прочитать изображение.")
        except (OSError, ValueError, cv2.error) as exc:
            self.show_error("Ошибка загрузки", str(exc))
            return
        self.original_image = image
        self._revision += 1
        self.clear_result()
        self.lbl_status.setText("Изображение загружено")

    def segment_image(self):
        if self.original_image is None or self._worker is not None or self._close_requested:
            return
        try:
            settings = self.current_settings()
            worker = ProcessingWorker(self.original_image, settings, self)
        except Exception as exc:
            self.show_error("Ошибка запуска", str(exc))
            return
        self.clear_result()
        self._active_revision = self._revision
        self._worker = worker
        worker.progress.connect(self.on_progress, QtCore.Qt.QueuedConnection)
        worker.result_ready.connect(self.on_result, QtCore.Qt.QueuedConnection)
        worker.failed.connect(self.on_error, QtCore.Qt.QueuedConnection)
        worker.finished.connect(self.on_finished, QtCore.Qt.QueuedConnection)
        self.lbl_status.setText("Подготовка к обработке…")
        self.update_controls()
        worker.start()

    @QtCore.pyqtSlot(str)
    def on_progress(self, message):
        if not self._close_requested and self._active_revision == self._revision:
            self.lbl_status.setText(message)

    @QtCore.pyqtSlot(object)
    def on_result(self, result: ProcessingResult):
        if self._close_requested or self._active_revision != self._revision:
            return
        if result.settings != self.current_settings():
            return
        self.result = result
        self.overlay_image = result.overlay_image_bgr
        self.lbl_count.setText(f"Объекты: {result.object_count}")
        self.lbl_time.setText(f"Время: {result.elapsed_seconds:.2f} с")
        self.lbl_status.setText("Готово")
        self.rb_seg.setChecked(True)
        self.update_view()

    @QtCore.pyqtSlot(str, str)
    def on_error(self, message, details):
        if not self._close_requested:
            self.lbl_status.setText("Ошибка обработки")
            self.show_error("Ошибка обработки", message, details)

    @QtCore.pyqtSlot()
    def on_finished(self):
        worker = self._worker
        self._worker = None
        self._active_revision = None
        if worker is not None:
            worker.deleteLater()
        self.update_controls()
        if self._close_requested:
            self.close()

    def save_image(self):
        if self.original_image is None or self._worker is not None or self._close_requested:
            return
        image = self.overlay_image if self.rb_seg.isChecked() else self.original_image
        if image is None:
            return
        filters = {"PNG (*.png)": ".png", "JPEG (*.jpg *.jpeg)": ".jpg", "BMP (*.bmp)": ".bmp"}
        filename, selected = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить файл", "", ";;".join(filters)
        )
        if not filename:
            return
        try:
            suffix = Path(filename).suffix.lower()
            if not suffix:
                suffix = filters.get(selected, ".png")
                filename += suffix
            if suffix not in {".png", ".jpg", ".jpeg", ".bmp"}:
                raise ValueError("Выберите формат PNG, JPEG или BMP.")
            success, buffer = cv2.imencode(suffix, image)
            if not success:
                raise ValueError("Не удалось закодировать изображение.")
            buffer.tofile(filename)
        except (OSError, ValueError, cv2.error) as exc:
            self.show_error("Ошибка сохранения", str(exc))

    def update_view(self):
        if self.original_image is None:
            return
        if self.rb_seg.isChecked() and self.overlay_image is not None:
            self.update_pixmap(self.overlay_image)
        else:
            self.update_pixmap(self.original_image)

    def update_pixmap(self, image):
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        qimage = QtGui.QImage(
            image.data, width, height, image.strides[0], QtGui.QImage.Format_BGR888
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        target = self.lbl_image.size()
        if target.width() and target.height():
            pixmap = pixmap.scaled(
                target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        self.lbl_image.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "rb_seg"):
            self.update_view()

    def closeEvent(self, event):
        if self._worker is not None:
            self._close_requested = True
            self._worker.requestInterruption()
            self.lbl_status.setText("Завершение текущего этапа перед закрытием…")
            self.update_controls()
            event.ignore()
            return
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
