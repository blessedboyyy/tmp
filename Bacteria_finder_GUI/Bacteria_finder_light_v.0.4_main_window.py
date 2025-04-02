from sys import path

path.append('../tmp')
path.append('../tmp/Bacteria_finder_GUI')
path.append('../tmp/Bacteria_finder_core_light')

from cv2 import IMREAD_COLOR, IMREAD_UNCHANGED, imdecode, imencode
from numpy import fromfile, uint8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap

from Bacteria_finder_core_light.BFmainlib import BF_image


class Ui_MainWindow(object):

    def __init__(self):
        self.segmentor = BF_image(verbose=False)
        self.original_bacteria_image = None
        self.shown_bacteria_image = None
        self.segmented_bacteria_image = None
        self.classified_bacteria_image = None
        self.sigma_changed = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 831)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        MainWindow.setFont(font)
        MainWindow.setWindowOpacity(3.0)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.ImageGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ImageGroupBox.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageGroupBox.sizePolicy().hasHeightForWidth())
        self.ImageGroupBox.setSizePolicy(sizePolicy)
        self.ImageGroupBox.setMaximumSize(QtCore.QSize(900, 900))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.ImageGroupBox.setFont(font)
        self.ImageGroupBox.setObjectName("ImageGroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.ImageGroupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ImageLabel = QtWidgets.QLabel(self.ImageGroupBox)
        self.ImageLabel.setText("")
        self.ImageLabel.setPixmap(QPixmap("Bacteria_finder_GUI/blank.png"))
        self.ImageLabel.setScaledContents(True)
        self.ImageLabel.setObjectName("ImageLabel")
        self.verticalLayout.addWidget(self.ImageLabel)
        self.horizontalLayout_2.addWidget(self.ImageGroupBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.ButtonsLayout = QtWidgets.QVBoxLayout()
        self.ButtonsLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.ButtonsLayout.setObjectName("ButtonsLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 15, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem1)
        self.LoadImageButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.LoadImageButton.setFont(font)
        self.LoadImageButton.setObjectName("LoadImageButton")
        self.ButtonsLayout.addWidget(self.LoadImageButton)
        spacerItem2 = QtWidgets.QSpacerItem(40, 15, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem2)
        self.SegmentImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.SegmentImageButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.SegmentImageButton.setFont(font)
        self.SegmentImageButton.setObjectName("SegmentImageButton")
        self.ButtonsLayout.addWidget(self.SegmentImageButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 15, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem3)
        self.SaveImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.SaveImageButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.SaveImageButton.setFont(font)
        self.SaveImageButton.setObjectName("SaveImageButton")
        self.ButtonsLayout.addWidget(self.SaveImageButton)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.ButtonsLayout.addItem(spacerItem4)
        self.ParametersgroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ParametersgroupBox.setEnabled(False)
        self.ParametersgroupBox.setMinimumSize(QtCore.QSize(0, 50))
        self.ParametersgroupBox.setObjectName("ParametersgroupBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.ParametersgroupBox)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 147, 26))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.SigmahorizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.SigmahorizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.SigmahorizontalLayout.setObjectName("SigmahorizontalLayout")
        self.Sigmalabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.Sigmalabel.setObjectName("Sigmalabel")
        self.SigmahorizontalLayout.addWidget(self.Sigmalabel)
        self.SigmadoubleSpinBox = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.SigmadoubleSpinBox.setDecimals(1)
        self.SigmadoubleSpinBox.setMaximum(20.0)
        self.SigmadoubleSpinBox.setSingleStep(0.1)
        self.SigmadoubleSpinBox.setProperty("value", 2.5)
        self.SigmadoubleSpinBox.setObjectName("SigmadoubleSpinBox")
        self.SigmahorizontalLayout.addWidget(self.SigmadoubleSpinBox)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.SigmahorizontalLayout.addItem(spacerItem5)
        self.ButtonsLayout.addWidget(self.ParametersgroupBox)
        self.CountertextBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.CountertextBrowser.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CountertextBrowser.sizePolicy().hasHeightForWidth())
        self.CountertextBrowser.setSizePolicy(sizePolicy)
        self.CountertextBrowser.setMinimumSize(QtCore.QSize(0, 30))
        self.CountertextBrowser.setMaximumSize(QtCore.QSize(16777215, 60))
        self.CountertextBrowser.setObjectName("CountertextBrowser")
        self.ButtonsLayout.addWidget(self.CountertextBrowser)
        self.ShowObjectsgroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ShowObjectsgroupBox.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowObjectsgroupBox.sizePolicy().hasHeightForWidth())
        self.ShowObjectsgroupBox.setSizePolicy(sizePolicy)
        self.ShowObjectsgroupBox.setMinimumSize(QtCore.QSize(0, 100))
        self.ShowObjectsgroupBox.setFlat(False)
        self.ShowObjectsgroupBox.setObjectName("ShowObjectsgroupBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.ShowObjectsgroupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 160, 51))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.ShowObjectsverticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.ShowObjectsverticalLayout.setContentsMargins(0, 0, 0, 0)
        self.ShowObjectsverticalLayout.setObjectName("ShowObjectsverticalLayout")
        self.OriginalradioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.OriginalradioButton.setObjectName("OriginalradioButton")
        self.ShowObjectsverticalLayout.addWidget(self.OriginalradioButton)
        self.SegmentedradioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.SegmentedradioButton.setObjectName("SegmentedradioButton")
        self.ShowObjectsverticalLayout.addWidget(self.SegmentedradioButton)
        self.ButtonsLayout.addWidget(self.ShowObjectsgroupBox)
        self.horizontalLayout_2.addLayout(self.ButtonsLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # All the connections are below
        # Load and Save
        self.LoadImageButton.clicked.connect(self.LoadImageButtonPushed)
        self.SaveImageButton.clicked.connect(self.SaveImageButtonPushed)

        # Segment image
        self.SegmentImageButton.clicked.connect(self.SegmentImageButtonPushed)
        self.SigmadoubleSpinBox.valueChanged.connect(self.SigmaValueChanged)

        # Change displayed objects by radiobuttons
        self.OriginalradioButton.toggled.connect(self.RadioButtonToggled)
        self.SegmentedradioButton.toggled.connect(self.RadioButtonToggled)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bacteria finder"))
        self.ImageGroupBox.setTitle(_translate("MainWindow", "Image"))
        self.LoadImageButton.setText(_translate("MainWindow", "Load image"))
        self.SegmentImageButton.setText(_translate("MainWindow", "Segment"))
        self.SaveImageButton.setText(_translate("MainWindow", "Save image"))
        self.ParametersgroupBox.setTitle(_translate("MainWindow", "Parameters"))
        self.Sigmalabel.setText(_translate("MainWindow", "Sigma:"))
        self.CountertextBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                    "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                    "p, li { white-space: pre-wrap; }\n"
                    "</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
                    "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9.75pt;\">Counter:</span></p>\n"
                    "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9.75pt;\">Objects = 0</span></p></body></html>"))
        self.ShowObjectsgroupBox.setTitle(_translate("MainWindow", "Show image"))
        self.OriginalradioButton.setText(_translate("MainWindow", "Original"))
        self.SegmentedradioButton.setText(_translate("MainWindow", "Segmented"))

    def RadioButtonToggled(self):
        '''
        Updates shown image by which radio button is pressed
        '''
        if self.OriginalradioButton.isChecked():
            self.shown_bacteria_image = self.original_bacteria_image.copy()
        elif self.SegmentedradioButton.isChecked():
            self.shown_bacteria_image = self.segmented_bacteria_image.copy()
        self.UpdateImage()
    
    def UpdateCounter(self):
        '''
        Updates on screen counter
        '''
        self.CountertextBrowser.setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:9.75pt; font-weight:400; font-style:normal;\">\n"
                                            "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Counter:</p>\n"
                                            f"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Objects = {len(self.segmentor.objects_db.keys())}</p>\n")
    
    def SigmaValueChanged(self):
        '''Function, that tracks whether the sigma value was changed'''
        self.sigma_changed = True

    
    def SegmentImageButtonPushed(self):
        '''
        Function that segments the image and displays the result
        '''

        if self.segmented_bacteria_image is None or self.sigma_changed:
            self.segmentor.load_image(image = self.original_bacteria_image.copy(), how = 'image')
            self.segmentor.preprocess_image()
            self.segmentor.segment_image(type='ridges', sigma=self.SigmadoubleSpinBox.value())
            self.segmented_bacteria_image = self.segmentor.segment_draw()[:,:,::-1].copy()
            self.CountertextBrowser.setEnabled(True)
            self.ShowObjectsgroupBox.setEnabled(True)
        if self.ShowObjectsgroupBox.isEnabled():
            self.SegmentedradioButton.setChecked(True)
        self.UpdateCounter()
        self.shown_bacteria_image = self.segmented_bacteria_image.copy()
        self.UpdateImage()
    
    def SaveImageButtonPushed(self):
        '''
        Function that opens save dialog window and saves the shown image
        '''
        # Getting the path for the image
        self.File_save_name = QtWidgets.QFileDialog.getSaveFileName(self.ImageLabel, 'Save file', '', "Image files (*.png *.jpg *.bmp *.gif)")
        self.File_save_path = self.File_save_name[0]

        # Check if the user didn't choose the path
        if self.File_save_path == '':
            return

        # Saving in choisen path
        _, buffed_image = imencode("." + self.File_save_path.split('/')[-1].split('.')[-1], self.shown_bacteria_image.copy())
        buffed_image.tofile(self.File_save_path)

    def LoadImageButtonPushed(self):
        '''
        Function that loads the path to an image
        '''
        # Loading the image
        self.File_load_name = QtWidgets.QFileDialog.getOpenFileName(self.ImageLabel, 'Open file', '', "Image files (*.png *.jpg *.bmp *.gif)")
        self.File_load_path = self.File_load_name[0]

        # Check if the user didn't choose the file
        if self.File_load_path == '':
            return

        # Calls the reverse function
        self.ReverseToOriginalImage()

    def ReverseToOriginalImage(self):
        '''
        Function, that reverses all changes made by user and reloads the image
        from previously chosen folder
        '''
        # Deleting previous results
        self.segmentor = BF_image(verbose=False)
        self.UpdateCounter()
        self.original_bacteria_image = None
        self.shown_bacteria_image = None
        self.segmented_bacteria_image = None
        self.classified_bacteria_image = None

        # Saving the image for future use in cv2
        self.shown_bacteria_image = imdecode(fromfile(self.File_load_path, dtype=uint8), IMREAD_COLOR)
        self.original_bacteria_image = self.shown_bacteria_image.copy()

        # Enabling and disabling widgets
        self.SegmentImageButton.setEnabled(True)
        self.SaveImageButton.setEnabled(True)
        self.LoadImageButton.setEnabled(True)
        self.ShowObjectsgroupBox.setEnabled(False)
        self.ParametersgroupBox.setEnabled(True)
        self.CountertextBrowser.setEnabled(False)

        # Displaying the image
        pixmap = QPixmap(self.File_load_path)
        self.ImageGroupBox.setMaximumSize(QtCore.QSize(int(900*self.shown_bacteria_image.shape[1]/self.shown_bacteria_image.shape[0]), 900))
        self.ImageLabel.setPixmap(QPixmap(pixmap))
    
    def UpdateImage(self, return_to_orig = False):
        '''
        Function, that updates image in the ImageLabel
        '''
            
        height, width = self.shown_bacteria_image.shape[0], self.shown_bacteria_image.shape[1]
        Q_displayed_image = QImage(self.shown_bacteria_image.data, width, height, width * 3, QImage.Format_RGB888).rgbSwapped()
        Q_displayed_image_pixmap = QPixmap.fromImage(Q_displayed_image)
        self.ImageLabel.setPixmap(Q_displayed_image_pixmap)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())