import cv2

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMainWindow, QColorDialog, QFileDialog

from GUI.Painter import Painter, convertQImageToMat, Img, ImageQt
from GUI.PaintsTensorFlowModul import PaintsTensorFlowModul


class Window(QMainWindow):
    def show(self):
        super().show()

    def __init__(self):
        super().__init__()
        uic.loadUi("./GUI/UI/ACGAN.ui", self)
        self.event_init()
        self.model = PaintsTensorFlowModul()

    def event_init(self):
        self.colorBtn.clicked.connect(self.color_btn_clicked)
        self.eBtn.clicked.connect(self.eraser_btn_clicked)
        self.panBtn.clicked.connect(self.pen_btn_clicked)
        self.runBtn.clicked.connect(self.run_btn_clicked)

        self.fileOpen.triggered.connect(self.file_open)
        self.fileOpen.setShortcut("Ctrl+O")
        self.fileSave.triggered.connect(self.file_save)
        self.fileSave.setShortcut("Ctrl+S")

        self.penSizeSlider.valueChanged.connect(
            lambda size: self.set_pan_size(size))

        self.penSize2.triggered.connect(
            lambda size: self.set_pan_size(2))
        self.penSize3.triggered.connect(
            lambda size: self.set_pan_size(3))
        self.penSize4.triggered.connect(
            lambda size: self.set_pan_size(4))
        self.penSize5.triggered.connect(
            lambda size: self.set_pan_size(5))
        self.penSize6.triggered.connect(
            lambda size: self.set_pan_size(6))
        self.penSize7.triggered.connect(
            lambda size: self.set_pan_size(7))

        self.painter = Painter()
        self.VBOX.addWidget(self.painter)

    def color_btn_clicked(self):
        def cliping(uint):
            if uint == 255:
                uint -= 1
            return uint

        color = QColorDialog.getColor()
        r, g, b, _ = color.getRgb()
        r = cliping(r)
        g = cliping(g)
        b = cliping(b)
        color = QColor(r, g, b)

        self.painter.set_pen_color(color)

    def set_pan_size(self, size=2):
        self.penSizeLabel.setText(str(size))
        self.painter.set_pensize(size=size)

    def eraser_btn_clicked(self):
        bg = self.painter.background
        self.painter.set_pen_color(QColor(bg, bg, bg))

    def pen_btn_clicked(self):
        self.penSizeLabel.setText(str(2))
        self.painter.setpen(pen_size=2, color=Qt.black)

    def run_btn_clicked(self):
        line = self.painter.line
        c_info = convertQImageToMat(self.painter.color_info)
        try:
            img = self.model.pred_image(line, c_info)
            self.imshow(title="result", img=img)
        except AttributeError:
            pass

    def imshow(self, img, title=""):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, img)

    def sketch_keras(self, path):
        line = self.model.sketch_keras.get_line(path)
        return line

    def file_open(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open File')
        if file_path is not "":
            line = self.sketch_keras(file_path)
            self.painter.line = line.copy()
            line = Img.fromarray(line)
            w, h = self.painter.painter_init(ImageQt(line))

            self.setFixedWidth(w + 20)
            self.setFixedHeight(h + 200)
        self.update()

    def file_save(self):
        # not_Implemented
        pass

