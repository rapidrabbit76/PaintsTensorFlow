from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor
from PyQt5.QtWidgets import QWidget, QLabel

from PIL.ImageQt import ImageQt
import PIL.Image as Img

from GUI.QtUtil import convertQImageToMat


class Painter(QWidget):
    background = 255

    def __init__(self):
        super().__init__()

        self.drawing = False
        self.last_point = QPoint()

        self.image = None
        self.line = None
        self.color_info = None
        self.setpen()
        self.pen_color = Qt.black
        self.pen_size = 2
        self.canvas = QLabel()
        self.canvas.setMouseTracking(True)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.color_info)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, e.pos())
            self.last_point = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, e):
        try:

            canvas_painter = QPainter(self)
            np_img = self.setpixmap(self.image, self.color_info)
            img = Img.fromarray(np_img)
            img = ImageQt(img)
            canvas_painter.drawPixmap(self.canvas.rect(),
                                      QPixmap.fromImage(img),
                                      self.canvas.rect())
        except TypeError:
            pass

    def setpixmap(self, img1, img2):
        img1 = convertQImageToMat(img1)
        img2 = convertQImageToMat(img2)
        img1[img2 != self.background] = img2[img2 != self.background]
        return img1

    def painter_init(self, image):
        self.image = image
        self.color_info = image.copy(image.rect())
        bg = self.background
        self.color_info.fill(QColor(bg, bg, bg))
        self.setFixedSize(self.image.size())
        self.canvas.setFixedSize(self.color_info.size())
        return self.width(), self.height()

    def set_pensize(self, size):
        self.setpen(pen_size=size, color=self.pen_color)

    def set_pen_color(self, color):
        self.setpen(pen_size=self.pen_size, color=color)

    def setpen(self, pen_size=2, color=Qt.black):

        self.pen = QPen(color,
                        pen_size,
                        Qt.SolidLine,
                        Qt.FlatCap,
                        Qt.RoundJoin)

        self.pen_size = pen_size
        self.pen_color = color
