import os
from functools import lru_cache

import cv2
import numpy as np
from typing import List

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QSpacerItem

from scripts.db.settings_manager import SettingsManager
from ui.widgets import BaseWidget


class ImageViewerWidget(BaseWidget):
    def prev_img_id(self):
        self.cur_img_id -= 1

    def next_img_id(self):
        self.cur_img_id += 1

    def none_cur_img_check(self):
        if self.cur_img is None:
            QMessageBox().warning(self, 'Warning', 'Не удается загрузить изображение/изображения')

    def load_images(self):
        file_filter = ' '.join([f'*.{pic_type}' for pic_type in SettingsManager().pic_types])
        filepaths = QFileDialog.getOpenFileNames(filter=f'Image files ({file_filter})')[0]
        self.image_paths = list(set(self.image_paths + filepaths))

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory()
        if not any(folder):
            return
        image_paths = []
        for image_name in os.listdir(folder):
            pic_type = image_name.split('.')[-1]
            if pic_type not in SettingsManager().pic_types:
                continue
            image_paths.append(os.path.join(folder, image_name))
        self.image_paths = image_paths

    def filename_row_changed(self, row):
        self.cur_img_id = row

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.last_global_cursor_pos = QCursor.pos()

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.last_global_cursor_pos = QCursor.pos()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.last_global_cursor_pos = QCursor.pos()

    def get_image_point(self) -> QPoint:
        pixmap_size = self.get_pixmap_size()
        image_point = self.imageLabel.mapFromGlobal(QCursor.pos())
        image_point.setX(min(max(image_point.x(), 0), pixmap_size[0]))
        image_point.setY(min(max(image_point.y(), 0), pixmap_size[1]))
        return image_point

    def get_actual_point(self, point: QPointF) -> QPoint:
        if self.cur_img is None:
            return QPoint()
        w, h = self.get_pixmap_size()
        return QPoint(int(point.x() * w), int(point.y() * h))

    def get_percentage_point(self, point: QPoint) -> QPointF:
        if self.cur_img is None:
            return QPointF()
        w, h = self.get_pixmap_size()
        return QPointF(point.x() / w, point.y() / h)

    def delete_cur_img(self):
        self.image_paths = self.image_paths[:self.cur_img_id] + self.image_paths[self.cur_img_id + 1:]

    def get_pixmap_size(self) -> tuple:
        if self.cur_img is None:
            return 0, 0
        h, w, _ = self.cur_img.shape
        if w > h:
            w_final = max(int(SettingsManager().percentage_img_size[0] / 100 * self.size().width()),
                          SettingsManager().min_size[0])
            h_final = int(h * w_final / w)
        else:
            h_final = max(int(SettingsManager().percentage_img_size[1] / 100 * self.size().height()),
                          SettingsManager().min_size[1])
            w_final = int(w * h_final / h)
        pixmap_size = (w_final, h_final)
        return pixmap_size

    def get_cur_pixmap(self) -> QPixmap:
        if self.cur_img is None:
            return QPixmap()

        pixmap_size = self.get_pixmap_size()

        img = cv2.resize(self.cur_img, pixmap_size)
        h, w, ch = img.shape
        qt_format = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        qt_format = qt_format.scaled(*pixmap_size, Qt.KeepAspectRatio)
        return QPixmap.fromImage(qt_format)

    def read_image(self, path: str):
        stream = open(path, "rb")
        numpyarray = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return img

    def write_image(self, path: str, image: np.ndarray):
        image_buf_arr = cv2.imencode('.' + path.split('.')[-1], image)[1]
        image_buf_arr.tofile(path)

    @property
    def image_paths(self):
        return self._image_paths

    @image_paths.setter
    def image_paths(self, value: List[str]):
        self._image_paths = list(map(lambda path: os.path.normpath(path), value))

        self.imageFilesList.clear()
        self.imageFilesList.addItems(map(lambda path: os.path.split(path)[-1], self._image_paths))

        # cur_img_id устанавливается после imageFilesList, потому что использует изменение индекса в imageFilesList
        self.cur_img_id = 0

    @property
    def cur_img_id(self):
        return self._cur_img_id

    @cur_img_id.setter
    def cur_img_id(self, value: int):
        if len(self._image_paths) == 0:
            self._cur_img_id = 0
            self._cur_img = None
        else:
            self._cur_img_id = value % len(self._image_paths)

            self._cur_img = self.read_image(self._image_paths[self._cur_img_id])
            if self._cur_img is None:
                self.imageLabel.setText('None')
            self._cur_img = cv2.cvtColor(self._cur_img, cv2.COLOR_BGR2RGB)

            self.none_cur_img_check()

            self.imageFilesList.setCurrentRow(self._cur_img_id)

    @property
    def cur_img(self):
        return self._cur_img

    @property
    def last_image_point(self):
        return self.imageLabel.mapFromGlobal(self.last_global_cursor_pos)

    @last_image_point.setter
    def last_image_point(self, val):
        self.last_global_cursor_pos = self.imageLabel.mapToGlobal(val)

    def __init__(self, parent, ui):
        super().__init__(parent, ui)

        self.last_global_cursor_pos = QCursor.pos()

        self._image_paths = []
        self._cur_img_id = 0
        self._cur_img = None

        self.prevButton.clicked.connect(self.prev_img_id)
        self.deleteButton.clicked.connect(self.delete_cur_img)
        self.nextButton.clicked.connect(self.next_img_id)

        self.imageFilesList.currentRowChanged.connect(self.filename_row_changed)

        self.loadImagesButton.clicked.connect(self.load_images)
        self.loadFolderButton.clicked.connect(self.load_folder)
