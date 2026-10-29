import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from Bacteria_finder_pipeline import (
    STAGES, GmmSettings, OtsuSettings, PipelineSettings, ProcessingResult, SlicSettings, available_methods,
)
from Bacteria_finder_pipeline.worker import ProcessingWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.overlay_image = None
        self.mask_image = None
        self.result = None
        self._worker = None
        self._revision = 0
        self._active_revision = None
        self._close_requested = False
        self._selected_segmentation = "light_ridges"
        self._method_stage_values = {}
        self.init_ui()

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle("Bacteria Finder — Light + IMBP")
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
        view_layout = QtWidgets.QHBoxLayout(gb_view)
        self.rb_orig = QtWidgets.QRadioButton("Оригинал")
        self.rb_seg = QtWidgets.QRadioButton("Сегмент")
        self.rb_mask = QtWidgets.QRadioButton("Маска")
        self.rb_orig.setChecked(True)
        view_layout.addWidget(self.rb_orig)
        view_layout.addWidget(self.rb_seg)
        view_layout.addWidget(self.rb_mask)
        self.rb_orig.toggled.connect(self.update_view)
        self.rb_seg.toggled.connect(self.update_view)
        self.rb_mask.toggled.connect(self.update_view)
        ctrl.addWidget(gb_view)

        gb_stages = QtWidgets.QGroupBox("Этапы обработки")
        gb_stages.setFont(QtGui.QFont("Times New Roman", 12))
        stages_layout = QtWidgets.QFormLayout(gb_stages)
        stages_layout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.stage_combos = {}
        defaults = PipelineSettings()
        for stage in STAGES:
            combo = QtWidgets.QComboBox()
            for method in available_methods(stage, defaults.segmentation):
                combo.addItem(method.name, method.id)
            combo.setCurrentIndex(combo.findData(getattr(defaults, stage.key)))
            combo.setToolTip(stage.description)
            if stage.key == "segmentation":
                combo.currentIndexChanged.connect(self.segmentation_changed)
            else:
                combo.currentIndexChanged.connect(self.settings_changed)
            self.stage_combos[stage.key] = combo
            stages_layout.addRow(f"{stage.name}:", combo)
        ctrl.addWidget(gb_stages)

        self.gb_light = QtWidgets.QGroupBox("Параметры Light")
        params_layout = QtWidgets.QHBoxLayout(self.gb_light)
        params_layout.addWidget(QtWidgets.QLabel("σ сглаживания:"))
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setRange(0.1, 20.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(defaults.sigma)
        self.spin_sigma.valueChanged.connect(self.settings_changed)
        params_layout.addWidget(self.spin_sigma)
        ctrl.addWidget(self.gb_light)

        self.gb_otsu = QtWidgets.QGroupBox("Параметры Оцу")
        self.gb_otsu.setFont(QtGui.QFont("Times New Roman", 12))
        otsu_layout = QtWidgets.QFormLayout(self.gb_otsu)
        self.combo_otsu_channel = QtWidgets.QComboBox()
        for title, channel in (("Выравнивание фона", "flat_gray"), ("Яркость", "gray"), ("Плотность красителей", "od")):
            self.combo_otsu_channel.addItem(title, channel)
        self.combo_otsu_channel.currentIndexChanged.connect(self.settings_changed)
        otsu_layout.addRow("Канал:", self.combo_otsu_channel)
        self.check_otsu_auto = QtWidgets.QCheckBox("Автоматический порог")
        self.check_otsu_auto.setChecked(True)
        self.check_otsu_auto.toggled.connect(self.settings_changed)
        otsu_layout.addRow(self.check_otsu_auto)
        self.spin_otsu_threshold = QtWidgets.QDoubleSpinBox()
        self.spin_otsu_threshold.setRange(0, 255)
        self.spin_otsu_threshold.setDecimals(1)
        self.spin_otsu_threshold.setValue(128)
        self.spin_otsu_threshold.valueChanged.connect(self.settings_changed)
        otsu_layout.addRow("Порог:", self.spin_otsu_threshold)
        ctrl.addWidget(self.gb_otsu)

        self.gb_slic = QtWidgets.QGroupBox("Параметры SLIC")
        self.gb_slic.setFont(QtGui.QFont("Times New Roman", 12))
        slic_layout = QtWidgets.QFormLayout(self.gb_slic)
        self.combo_slic_mode = QtWidgets.QComboBox()
        for title, mode in (("Random Forest", "rf"), ("Аналитический", "analytical"), ("Ручной", "manual")):
            self.combo_slic_mode.addItem(title, mode)
        self.combo_slic_mode.currentIndexChanged.connect(self.settings_changed)
        slic_layout.addRow("Подбор K и M:", self.combo_slic_mode)
        self.spin_slic_k = QtWidgets.QSpinBox()
        self.spin_slic_k.setRange(1, 100000)
        self.spin_slic_k.setValue(defaults.slic.k)
        self.spin_slic_k.valueChanged.connect(self.settings_changed)
        slic_layout.addRow("K (суперпиксели):", self.spin_slic_k)
        self.spin_slic_m = QtWidgets.QDoubleSpinBox()
        self.spin_slic_m.setRange(0.01, 1000)
        self.spin_slic_m.setDecimals(2)
        self.spin_slic_m.setValue(defaults.slic.m)
        self.spin_slic_m.valueChanged.connect(self.settings_changed)
        slic_layout.addRow("M (компактность):", self.spin_slic_m)
        tile_note = QtWidgets.QLabel("Тайлы ≈ 405×404; Оцу внутри SLIC — авто.")
        tile_note.setWordWrap(True)
        slic_layout.addRow(tile_note)
        ctrl.addWidget(self.gb_slic)

        self.gb_gmm = QtWidgets.QGroupBox("Параметры GMM")
        self.gb_gmm.setFont(QtGui.QFont("Times New Roman", 12))
        gmm_layout = QtWidgets.QFormLayout(self.gb_gmm)
        self.spin_gmm_components = QtWidgets.QSpinBox()
        self.spin_gmm_components.setRange(2, 8)
        self.spin_gmm_components.setValue(defaults.gmm.n_components)
        self.spin_gmm_components.valueChanged.connect(self.settings_changed)
        gmm_layout.addRow("Число кластеров:", self.spin_gmm_components)
        self.spin_gmm_min_area = QtWidgets.QSpinBox()
        self.spin_gmm_min_area.setRange(0, 100000)
        self.spin_gmm_min_area.setValue(defaults.gmm.min_area)
        self.spin_gmm_min_area.valueChanged.connect(self.settings_changed)
        gmm_layout.addRow("Мин. площадь, пикс.:", self.spin_gmm_min_area)
        self.check_gmm_bilateral = QtWidgets.QCheckBox("Сглаживание bilateral")
        self.check_gmm_bilateral.setChecked(defaults.gmm.bilateral)
        self.check_gmm_bilateral.toggled.connect(self.settings_changed)
        gmm_layout.addRow(self.check_gmm_bilateral)
        self.spin_gmm_beta = QtWidgets.QDoubleSpinBox()
        self.spin_gmm_beta.setRange(0, 50)
        self.spin_gmm_beta.setDecimals(2)
        self.spin_gmm_beta.setSingleStep(0.1)
        self.spin_gmm_beta.setValue(defaults.gmm.hmrf_beta)
        self.spin_gmm_beta.valueChanged.connect(self.settings_changed)
        gmm_layout.addRow("HMRF β (0 — выкл.):", self.spin_gmm_beta)
        ctrl.addWidget(self.gb_gmm)

        self.gb_auto = QtWidgets.QGroupBox("Автоматический выбор")
        self.gb_auto.setFont(QtGui.QFont("Times New Roman", 12))
        auto_layout = QtWidgets.QVBoxLayout(self.gb_auto)
        auto_note = QtWidgets.QLabel(
            "Выбор Оцу, SLIC или GMM для каждого участка ≈ 405×404 пикселей. "
            "Используются штатные параметры IMBP и RF-подбор SLIC. "
            "Настройки отдельных методов на «Авто» не влияют."
        )
        auto_note.setWordWrap(True)
        auto_layout.addWidget(auto_note)
        self.btn_auto_details = QtWidgets.QPushButton("Методы по участкам…")
        self.btn_auto_details.clicked.connect(self.show_auto_details)
        auto_layout.addWidget(self.btn_auto_details)
        ctrl.addWidget(self.gb_auto)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        gb_status = QtWidgets.QGroupBox("Состояние")
        status_layout = QtWidgets.QVBoxLayout(gb_status)
        self.lbl_count = QtWidgets.QLabel("Объекты: 0")
        self.lbl_time = QtWidgets.QLabel("Время: —")
        self.lbl_status = QtWidgets.QLabel("Загрузите изображение")
        self.lbl_status.setWordWrap(True)
        self.lbl_details = QtWidgets.QLabel()
        self.lbl_details.setFont(QtGui.QFont("Times New Roman", 12))
        self.lbl_details.setWordWrap(True)
        self.lbl_details.setTextFormat(QtCore.Qt.PlainText)
        status_layout.addWidget(self.lbl_count)
        status_layout.addWidget(self.lbl_time)
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.lbl_details)
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
            otsu=OtsuSettings(
                channel=self.combo_otsu_channel.currentData(),
                threshold=None if self.check_otsu_auto.isChecked() else self.spin_otsu_threshold.value(),
            ),
            slic=SlicSettings(
                mode=self.combo_slic_mode.currentData(),
                k=self.spin_slic_k.value(), m=self.spin_slic_m.value(),
            ),
            gmm=GmmSettings(
                n_components=self.spin_gmm_components.value(),
                min_area=self.spin_gmm_min_area.value(),
                bilateral=self.check_gmm_bilateral.isChecked(),
                hmrf_beta=self.spin_gmm_beta.value(),
            ),
        )

    def segmentation_changed(self):
        self._method_stage_values[self._selected_segmentation] = {
            key: combo.currentData() for key, combo in self.stage_combos.items()
            if key != "segmentation"
        }
        method = self.stage_combos["segmentation"].currentData()
        defaults = PipelineSettings(segmentation=method)
        saved = self._method_stage_values.get(method, {})
        for stage in STAGES:
            if stage.key == "segmentation":
                continue
            combo = self.stage_combos[stage.key]
            combo.blockSignals(True)
            combo.clear()
            for option in available_methods(stage, method):
                combo.addItem(option.name, option.id)
            index = combo.findData(saved.get(stage.key, getattr(defaults, stage.key)))
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)
        self._selected_segmentation = method
        self.settings_changed()

    def update_controls(self):
        busy = self._worker is not None or self._close_requested
        has_image = self.original_image is not None
        self.btn_load.setEnabled(not busy)
        self.btn_segment.setEnabled(has_image and not busy)
        self.btn_save.setEnabled(has_image and not busy)
        self.spin_sigma.setEnabled(not busy)
        for combo in self.stage_combos.values():
            combo.setEnabled(not busy and combo.count() > 1)
        method = self.stage_combos["segmentation"].currentData()
        self.gb_light.setVisible(method == "light_ridges")
        self.gb_otsu.setVisible(method == "imbp_otsu")
        self.gb_slic.setVisible(method == "imbp_slic")
        self.gb_gmm.setVisible(method == "imbp_gmm")
        self.gb_auto.setVisible(method == "imbp_auto")
        for control in (self.spin_gmm_components, self.spin_gmm_min_area, self.check_gmm_bilateral, self.spin_gmm_beta):
            control.setEnabled(not busy)
        self.btn_auto_details.setEnabled(
            not busy and method == "imbp_auto" and self.result is not None
            and bool(self.result.segmentation.tile_methods)
        )
        self.combo_otsu_channel.setEnabled(not busy)
        self.check_otsu_auto.setEnabled(not busy)
        self.spin_otsu_threshold.setEnabled(not busy and not self.check_otsu_auto.isChecked())
        self.combo_slic_mode.setEnabled(not busy)
        manual = self.combo_slic_mode.currentData() == "manual"
        self.spin_slic_k.setEnabled(not busy and manual)
        self.spin_slic_m.setEnabled(not busy and manual)
        self.rb_seg.setEnabled(self.result is not None and not busy)
        self.rb_mask.setVisible(method != "light_ridges")
        self.rb_mask.setEnabled(self.mask_image is not None and not busy)
        self.lbl_details.setVisible(bool(self.lbl_details.text()))
        self.progress.setVisible(self._worker is not None)

    def clear_result(self):
        self.result = None
        self.overlay_image = None
        self.mask_image = None
        label = "Объекты" if self.stage_combos["segmentation"].currentData() == "light_ridges" else "Компоненты маски"
        self.lbl_count.setText(f"{label}: 0")
        self.lbl_details.clear()
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

    def show_auto_details(self):
        if self._worker is not None or self._close_requested or self.result is None:
            return
        methods = self.result.segmentation.tile_methods
        if not methods:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Авто — методы по участкам")
        dialog.resize(620, 420)
        layout = QtWidgets.QVBoxLayout(dialog)
        note = QtWidgets.QLabel("Координаты пикселей от 1; обе границы включены.")
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QtWidgets.QTableWidget(len(methods), 4)
        table.setHorizontalHeaderLabels(("Участок", "X", "Y", "Метод"))
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        names = {"otsu": "Оцу", "slic": "SLIC", "gmm": "GMM"}
        for row, tile in enumerate(methods):
            y0, y1, x0, x1 = tile.bbox
            values = (str(row + 1), f"{x0 + 1}–{x1}", f"{y0 + 1}–{y1}", names[tile.method])
            for column, value in enumerate(values):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(table)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Close).setText("Закрыть")
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()
        dialog.deleteLater()

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
        mask = result.segmentation.mask
        self.mask_image = None if mask is None else mask.astype(np.uint8) * 255
        self.lbl_count.setText(f"{result.segmentation.count_label}: {result.object_count}")
        self.lbl_details.setText(result.segmentation.details)
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
        image = self.current_image()
        if image is None:
            return
        filters = {"PNG (*.png)": ".png", "JPEG (*.jpg *.jpeg)": ".jpg", "BMP (*.bmp)": ".bmp"}
        if self.rb_mask.isChecked():
            filters.pop("JPEG (*.jpg *.jpeg)")
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
            if self.rb_mask.isChecked() and suffix not in {".png", ".bmp"}:
                raise ValueError("Сохраните бинарную маску в PNG или BMP без потерь.")
            success, buffer = cv2.imencode(suffix, image)
            if not success:
                raise ValueError("Не удалось закодировать изображение.")
            buffer.tofile(filename)
        except (OSError, ValueError, cv2.error) as exc:
            self.show_error("Ошибка сохранения", str(exc))

    def current_image(self):
        if self.rb_mask.isChecked() and self.mask_image is not None:
            return self.mask_image
        if self.rb_seg.isChecked() and self.overlay_image is not None:
            return self.overlay_image
        return self.original_image

    def update_view(self):
        image = self.current_image()
        if image is not None:
            self.update_pixmap(image)

    def update_pixmap(self, image):
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        image_format = QtGui.QImage.Format_Grayscale8 if image.ndim == 2 else QtGui.QImage.Format_BGR888
        qimage = QtGui.QImage(
            image.data, width, height, image.strides[0], image_format
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
