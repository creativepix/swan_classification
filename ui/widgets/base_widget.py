import numpy as np
from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt


class BaseWidget(QWidget):
    def __init__(self, parent, ui):
        super().__init__(parent)

        self.ui_cls = ui
        self.ui_cls.setupUi(self, self)

        self.parent = parent

    def img2pixmap(self, img: np.array) -> QPixmap:
        h, w, ch = img.shape
        qt_format = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        qt_format = qt_format.scaled(*self.pixmap_size, Qt.KeepAspectRatio)
        return QPixmap.fromImage(qt_format)

    def retranslateUi(self, *args, **kwargs):
        self.ui_cls.retranslateUi(self, *args, **kwargs)
