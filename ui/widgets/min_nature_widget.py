from functools import lru_cache
import io
import math
import os
from enum import Enum
from typing import List, Tuple, Union
import zipfile

import cv2
import numpy as np
import torch
import pandas as pd

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt, QRect, QPoint, QLine, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QBrush, QCursor, QPolygon
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget, QInputDialog, QDialogButtonBox, QFormLayout, QLineEdit, \
    QDoubleSpinBox, QPushButton, QDialog, QSpinBox, QLabel, QListWidget, QComboBox

from scripts.db.settings_manager import SettingsManager
from ui.qtgeometry import QFigure
from ui.qtdesigner import Ui_UIInterface

from ui.widgets.image_viewer_widget import ImageViewerWidget

import ast
from transformers import CLIPModel, AutoProcessor
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import time
import pickle
from numpy import linalg
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
from copy import deepcopy
import pickle


mean_embeddings = {key: np.array(val) for key, val in pickle.load(open("./weights/mean_embedding.pkl", "rb")).items()}


print('loading models...')

t = time.time()
clip = CLIPModel.from_pretrained("./weights/vitl-336")
processor = AutoProcessor.from_pretrained("./weights/vitl-336")
print(f'clip loaded in {time.time() - t}')

t = time.time()
swan_yolo = YOLO("./weights/yolo/swan.pt")
head_yolo = YOLO("./weights/yolo/head.pt")
print(f'yolo swan loaded in {time.time() - t}')

t = time.time()
tabular_predictor_2clips = TabularPredictor(label='label').load("./weights/autogluon/autogluon_2clips")
full_autogluon = TabularPredictor(label='label').load("./weights/autogluon/full_autogluon")
head_autogluon = TabularPredictor(label='label').load("./weights/autogluon/head_autogluon")
swan_cropped_autogluon = TabularPredictor(label='label').load("./weights/autogluon/swan_cropped_autogluon")
print(f'tabulars loaded in {time.time() - t}')

tabular_label2cls = ['Малый', 'Кликун', 'Шипун']


# imgs - list of pillow images
def clip_encode(img, device):
    img = processor(images=img, return_tensors="pt")
    embeds = clip.get_image_features(img.pixel_values.to(device))
    if device == torch.device('cuda'):
        embeds = embeds.cpu().detach()
    embeds = embeds.numpy()[0]
    return embeds

def read_image(path: str):
    stream = open(path, "rb")
    numpyarray = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return img

#['Шипун', 'Кликун', 'Малый']
def predict(paths, device=torch.device('cpu')):
    is_cuda = torch.device('cuda') == device
    global clip, head_yolo, swan_yolo

    out_preds = []

    swan_yolo.to(device)
    head_yolo.to(device)
    clip = clip.to(device)

    for path in tqdm(paths):
        img = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(img)
        h_img, w_img, _ = img.shape

        results_swan = swan_yolo(pillow_img, verbose=False)
        results_heads = head_yolo(pillow_img, verbose=False)
        if results_heads[0].boxes is None:
            continue
        boxes = results_heads[0].boxes.xyxy

        all_head_embeddings = []
        for i, box in enumerate(boxes):
            if is_cuda:
                box = box.cpu().detach()
            box = box.numpy().astype(np.int32)

            head = deepcopy(img)[box[1]:box[3], box[0]:box[2]]
            head_embeddings = clip_encode(Image.fromarray(head), device)
            all_head_embeddings.append(head_embeddings)

        full_clip_embeddings = clip_encode(pillow_img, device)
        df_full_clip_embeddings = pd.DataFrame([full_clip_embeddings], columns=[f'embed_{i}' for i in range(len(full_clip_embeddings))])

        cls = full_autogluon.predict(df_full_clip_embeddings).values[0]
        full_autogluon_pred = tabular_label2cls[cls]

        best_sim = -1
        best_sim_key = None
        label2cls = {'whooper': 'Кликун', 'mute': 'Шипун', 'bewick': 'Малый'}
        for key, mean_embedding in mean_embeddings.items():
            sim = np.dot(full_clip_embeddings, mean_embedding) / (linalg.norm(full_clip_embeddings) * linalg.norm(mean_embedding))
            if sim > best_sim:
                best_sim = sim
                best_sim_key = key
        if best_sim_key is None:
            best_sim_key = list(label2cls.keys())[0]
        mean_pred = label2cls[best_sim_key]

        

        by_heads_pred = []
        clip_embeddings_full = clip_encode(pillow_img, device).tolist()
        if len(all_head_embeddings) != 0:
            clip_embeddings_head = np.array(all_head_embeddings).mean(axis=0).tolist()
            clip_embeddings = clip_embeddings_full + clip_embeddings_head

            df_clip_embeddings_head = pd.DataFrame([clip_embeddings_head], columns=[f'embed_{i}' for i in range(len(clip_embeddings_head))])
            by_heads_pred.append(head_autogluon.predict(df_clip_embeddings_head).values[0])

            tabular_pred = tabular_predictor_2clips.predict(pd.DataFrame([clip_embeddings], columns=[f'embed_{i}' for i in range(len(clip_embeddings))])).values[0]
            combined_autogluon_pred = tabular_label2cls[tabular_pred]
        else:
            combined_autogluon_pred = None

        if any(by_heads_pred):
            cls = max([(by_heads_pred.count(cls), cls) for cls in set(by_heads_pred)])[1]
            yolo_autogluon_head_pred = tabular_label2cls[cls]
        else:
            yolo_autogluon_head_pred = None

        #class_name = SettingsManager().labels[0]
        points = []

        classes = results_swan[0].boxes.cls
        if is_cuda:
            classes = classes.detach().cpu()
        classes = classes.numpy().astype(np.int32).tolist()
        if results_swan[0].masks is not None:
            masks_contours = results_swan[0].masks.xy
            masks = results_swan[0].masks.data

            for mask in masks:
                if is_cuda:
                    mask = mask.detach().cpu()
                mask = mask.numpy().astype(np.uint8)
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

                swan_cropped = cv2.bitwise_and(img, img, mask=mask)
                swan_cropped_embeddings = clip_encode(swan_cropped, device).tolist()
                df_swan_cropped_embeddings = pd.DataFrame([swan_cropped_embeddings], columns=[f'embed_{i}' for i in range(len(swan_cropped_embeddings))])
                cls = swan_cropped_autogluon.predict(df_swan_cropped_embeddings).values[0]
                swan_cropped_autogluon_pred = tabular_label2cls[cls]

            for swan_contours in masks_contours:
                points.append([[p[0] / w_img, p[1] / h_img] for p in swan_contours.astype(np.int32).tolist() ])
        cls = max([(classes.count(cls), cls) for cls in set(classes)])[1]
        yolo_swan_segmentation_pred = tabular_label2cls[cls]

        preds = [full_autogluon_pred, mean_pred, combined_autogluon_pred, yolo_swan_segmentation_pred, yolo_autogluon_head_pred, swan_cropped_autogluon_pred]
        new_preds = [pred for pred, koef in zip(preds, SettingsManager().pred_koefs) if koef and pred is not None]
        print(preds, new_preds)

        try:
            cls = max([(new_preds.count(cls), cls) for cls in set(new_preds)])[1]
        except IndexError:
            cls = 'Шипун'
        number = len(points)
        out_preds.append([path, cls, number, points])
    
    return out_preds


class DataView(Enum):
    NOTHING = 0
    BBOXES = 1
    POLYGONS = 2


class MinNatureWidget(ImageViewerWidget):
    def save_everything(self):
        filepath = QFileDialog.getSaveFileName(filter='CSV files (*.csv)')[0]
        if not any(filepath):
            return
        self.data.to_csv(filepath, index=False)

    def load_everything(self):
        filepath = QFileDialog.getOpenFileName(filter='CSV files (*.csv)')[0]
        if not any(filepath):
            return
        try:
            data = pd.read_csv(filepath)
        except pd.errors.ParserError:
            QMessageBox().warning(
                self, 'Warning', '''Что-то пошло не так при чтении файла''')
            return

        data.loc[:, 'points'] = [ast.literal_eval(data.loc[i, 'points']) for i in range(len(data))]
        self.data = data
        self.update_prediction()

    def save_csv(self):
        outpath = QFileDialog.getSaveFileName(filter='CSV files (*.csv)')[0]
        if not any(outpath):
            return
        data = []
        for i in range(len(self.data)):
            name, class_name, *_ = self.data.iloc[i]
            cls = 1 if class_name == 'Малый' else 2 if class_name == 'Кликун' else 3
            data.append([name.replace('\\', '/').split('/')[-1], cls])
        data = pd.DataFrame(data, columns=['name', 'class_x'])
        data.to_csv(outpath, index=False, sep=';')

    def predict_all(self):
        preds = predict(self.image_paths, device=torch.device('cuda' if self.use_cuda else 'cpu'))
        self.data = pd.DataFrame(preds, columns=['filepath', 'class_name', 'number', 'points'])
        self.update_prediction()

    def get_corrected_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def update_prediction(self):
        self.update_ui_prediction()

    def update_ui_prediction(self):
        if len(self.image_paths) == 0:
            self.predictionList.clear()
            return
        cur_preds = [self.predictionList.item(i).text() for i in range(self.predictionList.count())]
        cur_data = self.get_current_data()
        if cur_data.empty:
            self.predictionList.clear()
            return
        new_preds = cur_data[['class_name', 'number']].values.tolist()
        new_preds = [f'Тип: {pred[0]}; Численность: {pred[1]};' for pred in new_preds]
        if cur_preds == new_preds:
            return
        self.predictionList.clear()
        self.predictionList.addItems(new_preds)

    def use_cuda_changed(self, _):
        self.use_cuda = self.useCUDACheckbox.isChecked()

    def paintEvent(self, event):
        pixmap = self.get_cur_pixmap()

        figures = self.get_figures()
        if any(figures):
            qpainter = QPainter(pixmap)
            qpainter.setBrush(QBrush(QColor(0, 125, 125, 50), Qt.SolidPattern))
            for figure in figures:
                color = QColor(*SettingsManager().figure_color)
                qpainter.setPen(color)

                if self.data_view == DataView.POLYGONS:
                    figure = self.get_actual_figure(figure)
                    qpainter.drawPolygon(figure)
                elif self.data_view == DataView.BBOXES:
                    figure = self.get_actual_figure(figure)
                    x1, y1, x2, y2 = int(figure.left()), int(figure.top()), int(figure.right()), int(figure.bottom())
                    figure = QFigure([QPoint(x1, y1), QPoint(x2, y1), QPoint(x2, y2), QPoint(x1, y2)])
                    qpainter.drawPolygon(figure)
            qpainter.end()

        self.imageLabel.setPixmap(pixmap)
        if not pixmap.size().isNull():
            self.parent.setMinimumSize(int(pixmap.width() / (SettingsManager().percentage_img_size[0] / 100)),
                                       int(pixmap.height() / (SettingsManager().percentage_img_size[1] / 100)))

    def get_current_data(self) -> pd.DataFrame:
        if len(self.image_paths) == 0:
            return pd.DataFrame([], columns=self.data.columns)
        series = self.data[self.data['filepath'].str.replace("\\", "/", regex=True) == self.image_paths[self.cur_img_id].replace("\\", "/")]
        if series.empty:
            return pd.DataFrame([], columns=self.data.columns)
        return series

    def edit_row_settings(self, row: int):
        dialog = QDialog()
        dialog.setWindowTitle('Edit settings')
        dialog.setWindowModality(Qt.ApplicationModal)

        dialog.classLabel = 'Тип'
        dialog.classBox = QComboBox(dialog)
        dialog.classBox.addItems(SettingsManager().labels)
        dialog.classBox.setCurrentIndex(SettingsManager().labels.index(self.data.iloc[row]['class_name']))

        dialog.numberLabel = 'Численность'
        dialog.numberBox = QSpinBox(dialog)
        dialog.numberBox.setValue(self.data.iloc[row]['number'])

        dialog.layout = QFormLayout(dialog)
        dialog.layout.addRow(dialog.classLabel, dialog.classBox)
        dialog.layout.addRow(dialog.numberLabel, dialog.numberBox)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        dialog.layout.addWidget(buttonBox)
        if dialog.exec():
            self.data.loc[row, 'class_name'] = SettingsManager().labels[dialog.classBox.currentIndex()]
            self.data.loc[row, 'number'] = dialog.numberBox.value()
        self.update_prediction()

    def get_actual_figure(self, figure: QFigure) -> QFigure:
        if self.cur_img is None:
            return QFigure()
        w, h = self.get_pixmap_size()
        return QFigure([QPoint(int(point.x() * w), int(point.y() * h)) for point in figure.points])

    def get_figures(self) -> List[QFigure]:
        data = self.get_current_data()
        figures = [QFigure([QPointF(*point) for point in points]) for i in range(len(data)) for points in self.data.loc[data.index[i], 'points']]
        return figures

    def predictionListPressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.button() == Qt.RightButton:
            pos = self.predictionList.mapFromGlobal(QCursor.pos())
            row = self.predictionList.indexAt(pos).row()
            if row == -1:
                return
            self.edit_row_settings(row)
        else:
            QListWidget.mousePressEvent(self.predictionList, a0)

    def prediction_row_changed(self, row):
        self.predictionList.setCurrentRow(row)

    def filename_row_changed(self, row):
        super().filename_row_changed(row)
        self.update_prediction()

    def data_view_changed(self, index):
        self.data_view = DataView(index)

    def __init__(self, parent, ui=Ui_UIInterface):
        super().__init__(parent, ui)
        self.data = pd.DataFrame([], columns=['filepath', 'class_name', 'number', 'points'])

        self.use_cuda = torch.cuda.is_available()
        self.useCUDACheckbox.setChecked(self.use_cuda)
        self.useCUDACheckbox.setEnabled(torch.cuda.is_available())
        self.useCUDACheckbox.stateChanged.connect(self.use_cuda_changed)

        self.predictionList.mousePressEvent = self.predictionListPressEvent
        self.predictionList.currentRowChanged.connect(self.prediction_row_changed)

        self.data_view = DataView(0)
        self.dataViewComboBox.currentIndexChanged.connect(self.data_view_changed)

        self.predictAllButton.clicked.connect(self.predict_all)
        self.saveCsvButton.clicked.connect(self.save_csv)
        self.saveEverythingButton.clicked.connect(self.save_everything)
        self.loadEverythingButton.clicked.connect(self.load_everything)
