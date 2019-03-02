import cv2

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMainWindow, QColorDialog, QFileDialog, QMessageBox

from GUI.Painter import Painter, convertQImageToMat
from GUI.tensorModul.TensorModule import TensorModule
from GUI.QtUtil import err_message


class Window(QMainWindow):
    def show(self):
        super().show()

    def __init__(self):
        super().__init__()
        uic.loadUi("./GUI/src/UI/ACGAN.ui", self)
        self.__event_init()
        self.tensor_module = TensorModule()

    def __event_init(self):
        self.__liner_flag = False
        self.colorBtn.clicked.connect(self.__color_btn_clicked)
        self.eBtn.clicked.connect(self.__eraser_btn_clicked)
        self.panBtn.clicked.connect(self.__pen_btn_clicked)
        self.runBtn.clicked.connect(self.__run_btn_clicked)
        self.lineBtn.clicked.connect(self.__mkline)

        self.fileOpen.triggered.connect(self.__file_open)
        self.fileOpen.setShortcut("Ctrl+O")
        self.fileSave.triggered.connect(self.__file_save)
        self.fileSave.setShortcut("Ctrl+S")

        self.penSizeSlider.valueChanged.connect(
            lambda size: self.__set_pan_size(size))
        self.painter = Painter()
        self.VBOX.addWidget(self.painter)

    def __color_btn_clicked(self):
        def cliping(uint):
            if uint == 255:
                uint -= 2
            return uint

        rgb = list()
        color = QColorDialog.getColor()
        rgba = color.getRgb()

        for index in range(3):
            rgb.append(cliping(rgba[index]))
        color = QColor(rgb[0], rgb[1], rgb[2])

        self.painter.set_pen_color(color)

    def __set_pan_size(self, size=2):
        self.penSizeLabel.setText(str(size))
        self.painter.set_pensize(size=size)

    def __eraser_btn_clicked(self):
        bg = self.painter.background
        self.painter.set_pen_color(QColor(bg, bg, bg))

    def __pen_btn_clicked(self):
        self.penSizeLabel.setText(str(2))
        self.painter.setpen(pen_size=2, color=Qt.black)

    def __get_pred_image(self):
        line = self.painter.line
        c_info = convertQImageToMat(self.painter.color_info)
        try:
            img = self.tensor_module.pred_image(line, c_info)
        except AttributeError as e:
            img = None
            err_message(self, e.__str__())
        finally:
            return img

    def __run_btn_clicked(self):
        img = self.__get_pred_image()
        self.__show_image(title="result", img=img)

    def __show_image(self, img, title=""):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, img)

    def __mkline(self):
        if self.__liner_flag is False:
            try:
                image = self.painter.line
                line = self.tensor_module.get_line(image)
                self.painter.set_image(line)
                self.__liner_flag = True
            except Exception as e:
                err_message(self, e.__str__())
            finally:
                self.update()

    def __file_open(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   'Open File',
                                                   None,
                                                   self.tr("images (*.png *.jpeg *.jpg *.PNG *.JPEG *.JPG"))

        if file_path is "": return -1

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width = float(img.shape[1])
        height = float(img.shape[0])

        if width > height:
            rate = width / height
            new_height = 512
            new_width = int(512 * rate)
        else:
            rate = height / width
            new_width = 512
            new_height = int(rate * 512)

        img = cv2.resize(img,
                         (new_width, new_height),
                         cv2.INTER_LINEAR_EXACT)

        w, h = self.painter.set_image(img)
        self.setFixedWidth(w + 20)
        self.setFixedHeight(h + 200)
        self.update()
        self.__liner_flag = False

    def __file_save(self):
        image = self.__get_pred_image()

        if image is None: return -1

        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   'Save File',
                                                   None,
                                                   self.tr("images (*.png *.jpeg *.jpg *.PNG *.JPEG *.JPG"))
        if file_path is "": return -1

        result = QMessageBox.question(self,
                                      "Waifu2x",
                                      "Using Waifu2x upscale?",
                                      QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)

        if result == QMessageBox.Yes:
            image = self.tensor_module.upscale(image=image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, img=image)
