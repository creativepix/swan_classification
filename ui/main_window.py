from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QMainWindow

from scripts.db.settings_manager import SettingsManager
from ui.widgets import MinNatureWidget


class MainWindow(QMainWindow):
    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        rect = QRect(0, 0, a0.size().width(), a0.size().height())
        if self.widget_min_nature.geometry():
            self.widget_min_nature.setGeometry(rect)

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        self.widget_min_nature.keyPressEvent(a0)

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        self.widget_min_nature.keyReleaseEvent(a0)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.widget_min_nature.mousePressEvent(a0)

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.widget_min_nature.mouseMoveEvent(a0)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.widget_min_nature.mouseReleaseEvent(a0)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle(SettingsManager().main_window_title)

        self.widget_min_nature = MinNatureWidget(self)
        self.widget_min_nature.show()
        self.resize(self.widget_min_nature.size())
