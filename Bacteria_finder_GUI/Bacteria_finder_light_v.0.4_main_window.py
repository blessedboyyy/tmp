from sys import path

# Пути к модулям
path.append('../tmp')
path.append('../tmp/Bacteria_finder_GUI')
path.append('../tmp/Bacteria_finder_core_light')

from cv2 import IMREAD_COLOR, imdecode, imencode, drawContours
from numpy import fromfile, uint8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap

from Bacteria_finder_core_light.BFmainlib import BF_image

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.segmentor = BF_image(verbose=False)
        self.original_image = None
        self.overlay_image = None
        self.init_ui()

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.resize(1200, 800)
        font = QtGui.QFont("Times New Roman", 14)
        self.setFont(font)

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(central)

        # Область просмотра: изображение масштабируется под размер блока
        gb_image = QtWidgets.QGroupBox("Изображение")
        gb_image.setFont(font)
        vb_image = QtWidgets.QVBoxLayout(gb_image)
        self.lbl_image = QtWidgets.QLabel()
        self.lbl_image.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_image.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        vb_image.addWidget(self.lbl_image)
        layout.addWidget(gb_image, 3)

        # Панель управления
        ctrl = QtWidgets.QVBoxLayout()

        btn_load = QtWidgets.QPushButton("Загрузить изображение")
        btn_load.setFont(font)
        btn_load.clicked.connect(self.load_image)
        ctrl.addWidget(btn_load)

        self.btn_segment = QtWidgets.QPushButton("Сегментировать")
        self.btn_segment.setFont(font)
        self.btn_segment.setEnabled(False)
        self.btn_segment.clicked.connect(self.segment_image)
        ctrl.addWidget(self.btn_segment)

        self.btn_save = QtWidgets.QPushButton("Сохранить изображение")
        self.btn_save.setFont(font)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_image)
        ctrl.addWidget(self.btn_save)

        # Режим просмотра: оригинал или сегмент
        gb_view = QtWidgets.QGroupBox("Режим просмотра")
        gb_view.setFont(font)
        vb_view = QtWidgets.QVBoxLayout(gb_view)
        self.rb_orig = QtWidgets.QRadioButton("Оригинал")
        self.rb_seg = QtWidgets.QRadioButton("Сегмент")
        self.rb_orig.setFont(font)
        self.rb_seg.setFont(font)
        self.rb_orig.setChecked(True)
        vb_view.addWidget(self.rb_orig)
        vb_view.addWidget(self.rb_seg)
        ctrl.addWidget(gb_view)
        self.rb_orig.toggled.connect(self.update_view)
        self.rb_seg.toggled.connect(self.update_view)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress)

        gb_params = QtWidgets.QGroupBox("Параметры")
        gb_params.setFont(font)
        hl = QtWidgets.QHBoxLayout(gb_params)
        hl.addWidget(QtWidgets.QLabel("σ сглаживания:"))
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setRange(0.1, 20.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(2.5)
        self.spin_sigma.setFont(font)
        hl.addWidget(self.spin_sigma)
        ctrl.addWidget(gb_params)
        self.spin_sigma.valueChanged.connect(lambda _: setattr(self, 'overlay_image', None))

        gb_status = QtWidgets.QGroupBox("Состояние")
        gb_status.setFont(font)
        fs = QtWidgets.QFormLayout(gb_status)
        self.lbl_count = QtWidgets.QLabel("Объекты: 0")
        self.lbl_time = QtWidgets.QLabel("Время: —")
        fs.addRow(self.lbl_count)
        fs.addRow(self.lbl_time)
        ctrl.addWidget(gb_status)

        ctrl.addStretch()
        layout.addLayout(ctrl, 1)
        self.setCentralWidget(central)

    def load_image(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Открыть файл", "", "Изображения (*.png *.jpg *.bmp)")
        if not fn:
            return
        img = imdecode(fromfile(fn, dtype=uint8), IMREAD_COLOR)
        self.original_image = img
        self.overlay_image = None
        self.update_pixmap(self.original_image)
        self.btn_segment.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.lbl_count.setText("Объекты: 0")
        self.lbl_time.setText("Время: —")

    def segment_image(self):
        if self.original_image is None:
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        QtWidgets.QApplication.processEvents()
        import time; t0 = time.time()
        self.segmentor.load_image(how='image', image=self.original_image)
        self.segmentor.preprocess_image()
        self.segmentor.segment_image(type='ridges', sigma=self.spin_sigma.value())
        dt = time.time() - t0
        self.progress.setVisible(False)
        count = len(self.segmentor.objects_db)
        self.lbl_count.setText(f"Объекты: {count}")
        self.lbl_time.setText(f"Время: {dt:.2f} с")
        # создаём наложение контуров
        img = self.original_image.copy()
        contours = [obj.object_countour_coords for obj in self.segmentor.objects_db.values()]
        overlay = drawContours(img, contours, -1, (0, 255, 0), 1)
        self.overlay_image = overlay
        # если сегмент выбран, показываем его
        if self.rb_seg.isChecked():
            self.update_pixmap(self.overlay_image)

    def save_image(self):
        if self.rb_seg.isChecked() and self.overlay_image is not None:
            img = self.overlay_image
        else:
            img = self.original_image
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить файл", "", "Изображения (*.png *.jpg *.bmp)")
        if not fn:
            return
        fmt = fn.split('.')[-1]
        _, buf = imencode(f'.{fmt}', img)
        buf.tofile(fn)

    def update_view(self):
        # переключаем между оригиналом и сегментом
        if self.original_image is None:
            return
        if self.rb_seg.isChecked() and self.overlay_image is not None:
            self.update_pixmap(self.overlay_image)
        else:
            self.update_pixmap(self.original_image)

    def update_pixmap(self, img):
        # масштабируем изображение под lbl_image
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w*3, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        target = self.lbl_image.size()
        if target.width() and target.height():
            pix = pix.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.lbl_image.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # при изменении размера окна обновляем текущее изображение
        if self.rb_seg.isChecked() and self.overlay_image is not None:
            img = self.overlay_image
        else:
            img = self.original_image
        if img is not None:
            self.update_pixmap(img)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
