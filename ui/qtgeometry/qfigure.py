import math
from typing import List, Union, Tuple

from PyQt5.QtCore import QPoint, QPointF
from PyQt5.QtGui import QPolygon, QPolygonF

import numpy as np


class QFigure(QPolygonF):
    def left(self) -> float:
        points = self.tuple_points
        return min(map(lambda p: p[0], points))

    def right(self) -> float:
        points = self.tuple_points
        return max(map(lambda p: p[0], points))

    def top(self) -> float:
        points = self.tuple_points
        return min(map(lambda p: p[1], points))

    def bottom(self) -> float:
        points = self.tuple_points
        return max(map(lambda p: p[1], points))

    def width(self) -> float:
        return self.right() - self.left()

    def height(self) -> float:
        return self.bottom() - self.top()

    @property
    def points(self) -> List[QPointF]:
        return [self[i] for i in range(self.size())]

    @property
    def tuple_points(self) -> List[Tuple[float, float]]:
        return [(self[i].x(), self[i].y()) for i in range(self.size())]
